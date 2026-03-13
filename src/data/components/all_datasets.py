import os
import warnings
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from src.data.components.preprocessing_utils import preprocess

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)

# ─────────────────────────────────────────────────────────────────────────────
# MP-20
# ─────────────────────────────────────────────────────────────────────────────

class MP20(InMemoryDataset):
    """The MP20 dataset from Materials Project, as a PyG InMemoryDataset.

    In order to create a torch_geometric.data.InMemoryDataset, you need to implement four fundamental methods:
    - InMemoryDataset.raw_file_names(): A list of files in the raw_dir which needs to be found in order to skip the download.
    - InMemoryDataset.processed_file_names(): A list of files in the processed_dir which needs to be found in order to skip the processing.
    - InMemoryDataset.download(): Downloads raw data into raw_dir.
    - InMemoryDataset.process(): Processes raw data and saves it into the processed_dir.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["all.csv"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["mp20.pt"]

    def download(self) -> None:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="chaitjo/MP20_ADiT",
            filename="raw/all.csv",
            repo_type="dataset",
            local_dir=self.root,
        )

    def process(self) -> None:
        if os.path.exists(os.path.join(self.root, "raw/all.pt")):
            cached_data = torch.load(os.path.join(self.root, "raw/all.pt"))
        else:
            cached_data = preprocess(
                os.path.join(self.root, "raw/all.csv"),
                niggli=True,
                primitive=False,
                graph_method="crystalnn",
                prop_list=["formation_energy_per_atom"],
                use_space_group=True,
                tol=0.1,
                num_workers=32,
            )
            torch.save(cached_data, os.path.join(self.root, "raw/all.pt"))

        data_list = []
        for data_dict in cached_data:
            # extract attributes from data_dict
            graph_arrays = data_dict["graph_arrays"]
            atom_types = graph_arrays["atom_types"]
            frac_coords = graph_arrays["frac_coords"]
            cell = graph_arrays["cell"]
            lattices = graph_arrays["lattices"]
            lengths = graph_arrays["lengths"]
            angles = graph_arrays["angles"]
            num_atoms = graph_arrays["num_atoms"]

            # normalize the lengths of lattice vectors, which makes
            # lengths for materials of different sizes at same scale
            _lengths = lengths / float(num_atoms) ** (1 / 3)
            # convert angles of lattice vectors to be in radians
            _angles = np.radians(angles)
            # add scaled lengths and angles to graph arrays
            graph_arrays["length_scaled"] = _lengths
            graph_arrays["angles_radians"] = _angles
            graph_arrays["lattices_scaled"] = np.concatenate([_lengths, _angles])

            data = Data(
                id=data_dict["mp_id"],
                atom_types=torch.LongTensor(atom_types),
                frac_coords=torch.Tensor(frac_coords),
                cell=torch.Tensor(cell).unsqueeze(0),
                lattices=torch.Tensor(lattices).unsqueeze(0),
                lattices_scaled=torch.Tensor(graph_arrays["lattices_scaled"]).unsqueeze(0),
                lengths=torch.Tensor(lengths).view(1, -1),
                lengths_scaled=torch.Tensor(graph_arrays["length_scaled"]).view(1, -1),
                angles=torch.Tensor(angles).view(1, -1),
                angles_radians=torch.Tensor(graph_arrays["angles_radians"]).view(1, -1),
                num_atoms=torch.LongTensor([num_atoms]),
                num_nodes=torch.LongTensor([num_atoms]),  # special attribute used for PyG batching
                token_idx=torch.arange(num_atoms),
                dataset_idx=torch.tensor(
                    [0], dtype=torch.long
                ),  # 0 --> indicates periodic/crystal
            )
            # 3D coordinates (NOTE do not zero-center prior to graph construction)
            data.pos = torch.einsum(
                "bi,bij->bj",
                data.frac_coords,
                torch.repeat_interleave(data.cell, data.num_atoms, dim=0),
            )
            # space group number
            data.spacegroup = torch.LongTensor([data_dict["spacegroup"]])

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, os.path.join(self.root, "processed/mp20.pt"))


# ─────────────────────────────────────────────────────────────────────────────
# Perov-5
# ─────────────────────────────────────────────────────────────────────────────

class Perov5(InMemoryDataset):
    """Perov-5 perovskite dataset from CDVAE, in ADiT PyG format.

    Prepare data once before first use:
        1. Concatenate DiffCSP/CDVAE split CSVs into raw/all.csv:
               python other_datasets.py --data_dir /path/to/perov_5
           (or manually: cat train.csv val.csv test.csv > raw/all.csv, keeping
            the header only once)
        2. First dataset instantiation will run preprocess() and cache raw/all.pt.

    Standard splits (set in Perov5DataModule):
        train: [:18000]
        val:   [18000:20000]
        test:  [20000:]
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["all.csv"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["perov5.pt"]

    def download(self) -> None:
        raise RuntimeError(
            f"Perov-5 raw data not found at {self.raw_dir}/all.csv\n\n"
            f"Setup steps:\n"
            f"  1. Get CSVs from CDVAE repo (https://github.com/txie-93/cdvae/tree/main/data/perov_5)\n"
            f"     or DiffCSP repo (https://github.com/jiaor17/DiffCSP/tree/main/data/perov_5)\n"
            f"  2. Concatenate splits into raw/all.csv:\n"
            f"         python src/data/components/other_datasets.py --data_dir /path/to/perov_5\n"
            f"     This writes raw/all.csv and prints the train_end/val_end indices.\n"
        )

    def process(self) -> None:
        cached_pt = os.path.join(self.root, "raw/all.pt")
        raw_csv   = os.path.join(self.root, "raw/all.csv")

        if os.path.exists(cached_pt):
            cached_data = torch.load(cached_pt)
        else:
            cached_data = preprocess(
                raw_csv,
                niggli=True,
                primitive=False,
                graph_method="crystalnn",
                prop_list=["heat_ref"],
                use_space_group=True,
                tol=0.1,
                num_workers=32,
            )
            torch.save(cached_data, cached_pt)

        data_list = []
        for data_dict in cached_data:
            graph_arrays = data_dict["graph_arrays"]
            atom_types  = graph_arrays["atom_types"]
            frac_coords = graph_arrays["frac_coords"]
            cell        = graph_arrays["cell"]
            lattices    = graph_arrays["lattices"]
            lengths     = graph_arrays["lengths"]
            angles      = graph_arrays["angles"]
            num_atoms   = graph_arrays["num_atoms"]

            _lengths = lengths / float(num_atoms) ** (1 / 3)
            _angles  = np.radians(angles)
            graph_arrays["length_scaled"]   = _lengths
            graph_arrays["angles_radians"]  = _angles
            graph_arrays["lattices_scaled"] = np.concatenate([_lengths, _angles])

            data = Data(
                id=data_dict.get("material_id", data_dict.get("mp_id", "")),
                atom_types=torch.LongTensor(atom_types),
                frac_coords=torch.Tensor(frac_coords),
                cell=torch.Tensor(cell).unsqueeze(0),
                lattices=torch.Tensor(lattices).unsqueeze(0),
                lattices_scaled=torch.Tensor(graph_arrays["lattices_scaled"]).unsqueeze(0),
                lengths=torch.Tensor(lengths).view(1, -1),
                lengths_scaled=torch.Tensor(graph_arrays["length_scaled"]).view(1, -1),
                angles=torch.Tensor(angles).view(1, -1),
                angles_radians=torch.Tensor(graph_arrays["angles_radians"]).view(1, -1),
                num_atoms=torch.LongTensor([num_atoms]),
                num_nodes=torch.LongTensor([num_atoms]),
                token_idx=torch.arange(num_atoms),
                dataset_idx=torch.tensor([0], dtype=torch.long),
            )
            data.pos = torch.einsum(
                "bi,bij->bj",
                data.frac_coords,
                torch.repeat_interleave(data.cell, data.num_atoms, dim=0),
            )
            data.spacegroup = torch.LongTensor([data_dict["spacegroup"]])

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, os.path.join(self.root, "processed/perov5.pt"))


# ─────────────────────────────────────────────────────────────────────────────
# Carbon-24
# ─────────────────────────────────────────────────────────────────────────────

class Carbon24(InMemoryDataset):
    """Carbon-24 dataset from CDVAE, in ADiT PyG format.

    Prepare data once before first use:
        python src/data/components/other_datasets.py --data_dir /path/to/carbon_24

    Standard splits (set in Carbon24DataModule):
        train: [:10153]
        val:   [10153:11286]
        test:  [11286:]
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["all.csv"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["carbon24.pt"]

    def download(self) -> None:
        raise RuntimeError(
            f"Carbon-24 raw data not found at {self.raw_dir}/all.csv\n\n"
            f"Setup steps:\n"
            f"  1. Get CSVs from CDVAE repo (https://github.com/txie-93/cdvae/tree/main/data/carbon_24)\n"
            f"  2. Concatenate splits into raw/all.csv:\n"
            f"         python src/data/components/other_datasets.py --data_dir /path/to/carbon_24\n"
        )

    def process(self) -> None:
        cached_pt = os.path.join(self.root, "raw/all.pt")
        raw_csv   = os.path.join(self.root, "raw/all.csv")

        if os.path.exists(cached_pt):
            cached_data = torch.load(cached_pt)
        else:
            cached_data = preprocess(
                raw_csv,
                niggli=True,
                primitive=False,
                graph_method="crystalnn",
                prop_list=["energy_per_atom"],
                use_space_group=True,
                tol=0.1,
                num_workers=32,
            )
            torch.save(cached_data, cached_pt)

        data_list = []
        for data_dict in cached_data:
            graph_arrays = data_dict["graph_arrays"]
            atom_types  = graph_arrays["atom_types"]
            frac_coords = graph_arrays["frac_coords"]
            cell        = graph_arrays["cell"]
            lattices    = graph_arrays["lattices"]
            lengths     = graph_arrays["lengths"]
            angles      = graph_arrays["angles"]
            num_atoms   = graph_arrays["num_atoms"]

            _lengths = lengths / float(num_atoms) ** (1 / 3)
            _angles  = np.radians(angles)
            graph_arrays["length_scaled"]   = _lengths
            graph_arrays["angles_radians"]  = _angles
            graph_arrays["lattices_scaled"] = np.concatenate([_lengths, _angles])

            data = Data(
                id=data_dict.get("material_id", data_dict.get("mp_id", "")),
                atom_types=torch.LongTensor(atom_types),
                frac_coords=torch.Tensor(frac_coords),
                cell=torch.Tensor(cell).unsqueeze(0),
                lattices=torch.Tensor(lattices).unsqueeze(0),
                lattices_scaled=torch.Tensor(graph_arrays["lattices_scaled"]).unsqueeze(0),
                lengths=torch.Tensor(lengths).view(1, -1),
                lengths_scaled=torch.Tensor(graph_arrays["length_scaled"]).view(1, -1),
                angles=torch.Tensor(angles).view(1, -1),
                angles_radians=torch.Tensor(graph_arrays["angles_radians"]).view(1, -1),
                num_atoms=torch.LongTensor([num_atoms]),
                num_nodes=torch.LongTensor([num_atoms]),
                token_idx=torch.arange(num_atoms),
                dataset_idx=torch.tensor([0], dtype=torch.long),
            )
            data.pos = torch.einsum(
                "bi,bij->bj",
                data.frac_coords,
                torch.repeat_interleave(data.cell, data.num_atoms, dim=0),
            )
            data.spacegroup = torch.LongTensor([data_dict["spacegroup"]])

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, os.path.join(self.root, "processed/carbon24.pt"))


# ─────────────────────────────────────────────────────────────────────────────
# MPTS-52
# ─────────────────────────────────────────────────────────────────────────────

class MPTS52(InMemoryDataset):
    """MPTS-52 dataset from DiffCSP, in ADiT PyG format.

    Prepare data once before first use:
        python src/data/components/other_datasets.py --data_dir /path/to/mpts_52

    Standard splits (set in MPTS52DataModule):
        train: [:40328]
        val:   [40328:45369]
        test:  [45369:]
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["all.csv"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["mpts52.pt"]

    def download(self) -> None:
        raise RuntimeError(
            f"MPTS-52 raw data not found at {self.raw_dir}/all.csv\n\n"
            f"Setup steps:\n"
            f"  1. Get CSVs from DiffCSP repo (https://github.com/jiaor17/DiffCSP/tree/main/data/mpts_52)\n"
            f"  2. Concatenate splits into raw/all.csv:\n"
            f"         python src/data/components/other_datasets.py --data_dir /path/to/mpts_52\n"
        )

    def process(self) -> None:
        cached_pt = os.path.join(self.root, "raw/all.pt")
        raw_csv   = os.path.join(self.root, "raw/all.csv")

        if os.path.exists(cached_pt):
            cached_data = torch.load(cached_pt)
        else:
            cached_data = preprocess(
                raw_csv,
                niggli=True,
                primitive=False,
                graph_method="crystalnn",
                prop_list=["formation_energy_per_atom"],
                use_space_group=True,
                tol=0.1,
                num_workers=32,
            )
            torch.save(cached_data, cached_pt)

        data_list = []
        for data_dict in cached_data:
            graph_arrays = data_dict["graph_arrays"]
            atom_types  = graph_arrays["atom_types"]
            frac_coords = graph_arrays["frac_coords"]
            cell        = graph_arrays["cell"]
            lattices    = graph_arrays["lattices"]
            lengths     = graph_arrays["lengths"]
            angles      = graph_arrays["angles"]
            num_atoms   = graph_arrays["num_atoms"]

            _lengths = lengths / float(num_atoms) ** (1 / 3)
            _angles  = np.radians(angles)
            graph_arrays["length_scaled"]   = _lengths
            graph_arrays["angles_radians"]  = _angles
            graph_arrays["lattices_scaled"] = np.concatenate([_lengths, _angles])

            data = Data(
                id=data_dict.get("material_id", data_dict.get("mp_id", "")),
                atom_types=torch.LongTensor(atom_types),
                frac_coords=torch.Tensor(frac_coords),
                cell=torch.Tensor(cell).unsqueeze(0),
                lattices=torch.Tensor(lattices).unsqueeze(0),
                lattices_scaled=torch.Tensor(graph_arrays["lattices_scaled"]).unsqueeze(0),
                lengths=torch.Tensor(lengths).view(1, -1),
                lengths_scaled=torch.Tensor(graph_arrays["length_scaled"]).view(1, -1),
                angles=torch.Tensor(angles).view(1, -1),
                angles_radians=torch.Tensor(graph_arrays["angles_radians"]).view(1, -1),
                num_atoms=torch.LongTensor([num_atoms]),
                num_nodes=torch.LongTensor([num_atoms]),
                token_idx=torch.arange(num_atoms),
                dataset_idx=torch.tensor([0], dtype=torch.long),
            )
            data.pos = torch.einsum(
                "bi,bij->bj",
                data.frac_coords,
                torch.repeat_interleave(data.cell, data.num_atoms, dim=0),
            )
            data.spacegroup = torch.LongTensor([data_dict["spacegroup"]])

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, os.path.join(self.root, "processed/mpts52.pt"))


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build raw/all.csv from the original DiffCSP split CSVs
# ─────────────────────────────────────────────────────────────────────────────

def build_all_csv(data_dir: str) -> tuple:
    """Concatenate train/val/test CSVs into raw/all.csv.

    The DiffCSP/CDVAE split CSVs (train.csv, val.csv, test.csv) are used
    directly — no format conversion is needed. preprocess() reads the 'cif'
    column directly from these files.

    Order is strictly preserved: train rows first, then val, then test —
    so slicing by train_end / val_end gives the correct splits.

    Args:
        data_dir: Directory containing train.csv, val.csv, test.csv

    Returns:
        (train_end, val_end) — paste these into your datamodule config YAML
    """
    import pandas as pd

    dfs, sizes = [], {}
    for split in ["train", "val", "test"]:
        path = os.path.join(data_dir, f"{split}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)
            sizes[split] = len(df)
            print(f"  {split}: {len(df)} rows")
        else:
            print(f"  WARNING: {path} not found, skipping")

    all_df = pd.concat(dfs, ignore_index=True)

    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, "all.csv")
    all_df.to_csv(out_path, index=False)

    train_end = sizes.get("train", 0)
    val_end   = train_end + sizes.get("val", 0)

    print(f"\nWrote {len(all_df)} total rows → {out_path}")
    print(f"\nPaste into your datamodule config YAML:")
    print(f"  train_end: {train_end}")
    print(f"  val_end:   {val_end}")
    return train_end, val_end


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Concatenate DiffCSP split CSVs into raw/all.csv for ADiT."
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Directory containing train.csv, val.csv, test.csv (DiffCSP format)"
    )
    args = parser.parse_args()
    build_all_csv(args.data_dir)