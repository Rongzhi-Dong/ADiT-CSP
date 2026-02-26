"""Copyright (c) Meta Platforms, Inc. and affiliates."""
DEBUG_LDM_LATENT = False
import copy
import os
import random
import time
from typing import Any, Dict, Literal, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch.nn import ModuleDict
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torchmetrics import MeanMetric
from tqdm import tqdm

import wandb
from src.eval.crystal_generation import CrystalGenerationEvaluator
from src.eval.mof_generation import MOFGenerationEvaluator
from src.eval.molecule_generation import MoleculeGenerationEvaluator
from src.models.components.kabsch_utils import random_rotation_matrix
from src.models.vae_module import VariationalAutoencoderLitModule
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


IDX_TO_DATASET = {
    0: "mp20",
    1: "qm9",
    2: "qmof150",
}
DATASET_TO_IDX = {
    "mp20": 0,  # periodic
    "qm9": 1,  # non-periodic
    "qmof150": 0,  # periodic
}


class LatentDiffusionLitModule(LightningModule):
    """LightningModule for latent diffusion generative modellling of 3D atomic systems.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        autoencoder_ckpt: str,
        denoiser: torch.nn.Module,
        interpolant: DictConfig,
        augmentations: DictConfig,
        sampling: DictConfig,
        conditioning: DictConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: str,
        compile: bool,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # autoencoder models (first-stage model)
        self.autoencoder_ckpt = autoencoder_ckpt
        log.info(f"Loading Autoencoder ckpt: {autoencoder_ckpt}")
        self.autoencoder = VariationalAutoencoderLitModule.load_from_checkpoint(
            autoencoder_ckpt, map_location="cpu"
        )
        # freeze autoencoder
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()

        # denoiser model (second-stage model)
        self.denoiser = denoiser

        # interpolant for diffusion or flow matching training/sampling
        self.interpolant = interpolant

        # evaluator objects for computing metrics
        # self.val_generation_evaluators = {
        #     "mp20": CrystalGenerationEvaluator(
        #         dataset_cif_list=pd.read_csv(
        #             os.path.join(self.hparams.sampling.data_dir, f"mp_20/raw/all.csv")
        #         )["cif"].tolist()
        #     ),
        #     "qm9": MoleculeGenerationEvaluator(
        #         dataset_smiles_list=torch.load(
        #             os.path.join(self.hparams.sampling.data_dir, f"qm9/smiles.pt"),
        #         ),
        #         removeHs=self.hparams.sampling.removeHs,
        #     ),
        #     "qmof150": MOFGenerationEvaluator(),
        # }
        # self.test_generation_evaluators = copy.deepcopy(self.val_generation_evaluators)

        ### modified by RZD
        # --- START: robust evaluator setup (only create QM9 evaluator if qm9 enabled) ---
        # Helper to check if QM9 dataset is active in datamodule config
        qm9_enabled = False
        try:
            qm9_cfg = self.hparams.data.datamodule.datasets.get("qm9", None)
            if qm9_cfg is not None:
                # If proportion exists and >0 or explicit flag present, treat as enabled
                proportion = qm9_cfg.get("proportion", 0.0)
                qm9_enabled = float(proportion) > 0.0
        except Exception:
            qm9_enabled = False

        # mp20 evaluator (keep as before)
        mp20_eval = CrystalGenerationEvaluator(
            dataset_cif_list=pd.read_csv(
                os.path.join(self.hparams.sampling.data_dir, f"mp_20/raw/all.csv")
            )["cif"].tolist()
        )

        # qm9 evaluator: only try to load SMILES if qm9 is enabled
        if qm9_enabled:
            qm9_smiles_path = os.path.join(self.hparams.sampling.data_dir, "qm9", "smiles.pt")
            try:
                qm9_smiles = torch.load(qm9_smiles_path)
                log.info(f"Loaded QM9 smiles list from {qm9_smiles_path} (len={len(qm9_smiles)})")
            except FileNotFoundError:
                qm9_smiles = []
                log.warning(f"QM9 smiles file not found at {qm9_smiles_path}. Continuing with empty list.")
            except Exception as e:
                qm9_smiles = []
                log.warning(f"Unable to load QM9 smiles from {qm9_smiles_path}: {e}. Continuing with empty list.")

            qm9_eval = MoleculeGenerationEvaluator(
                dataset_smiles_list=qm9_smiles, removeHs=self.hparams.sampling.removeHs
            )
        else:
            # Create an empty MoleculeGenerationEvaluator or None — using empty evaluator keeps downstream code simple
            qm9_eval = MoleculeGenerationEvaluator(dataset_smiles_list=[], removeHs=self.hparams.sampling.removeHs)
            log.info("QM9 dataset disabled in config (proportion=0). QM9 evaluator created with empty smiles list.")

        # qmof150: keep original behavior (or similarly guard if you won't use it)
        qmof_eval = MOFGenerationEvaluator()

        self.val_generation_evaluators = {
            "mp20": mp20_eval,
            "qm9": qm9_eval,
            "qmof150": qmof_eval,
        }
        self.test_generation_evaluators = copy.deepcopy(self.val_generation_evaluators)
        # --- END: robust evaluator setup ---
                


        # metric objects for calculating and averaging across batches
        self.train_metrics = ModuleDict(
            {
                "loss": MeanMetric(),
                "x_loss": MeanMetric(),
                "x_loss t=[0,25)": MeanMetric(),
                "x_loss t=[25,50)": MeanMetric(),
                "x_loss t=[50,75)": MeanMetric(),
                "x_loss t=[75,100)": MeanMetric(),
                "t_avg": MeanMetric(),
                "dataset_idx": MeanMetric(),
            }
        )
        self.val_metrics = ModuleDict(
            {
                "mp20": ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "x_loss": MeanMetric(),
                        "x_loss t=[0,25)": MeanMetric(),
                        "x_loss t=[25,50)": MeanMetric(),
                        "x_loss t=[50,75)": MeanMetric(),
                        "x_loss t=[75,100)": MeanMetric(),
                        "t_avg": MeanMetric(),
                        "valid_rate": MeanMetric(),
                        "struct_valid_rate": MeanMetric(),
                        "comp_valid_rate": MeanMetric(),
                        "unique_rate": MeanMetric(),
                        "novel_rate": MeanMetric(),
                        "sampling_time": MeanMetric(),
                    }
                ),
                "qm9": ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "x_loss": MeanMetric(),
                        "x_loss t=[0,25)": MeanMetric(),
                        "x_loss t=[25,50)": MeanMetric(),
                        "x_loss t=[50,75)": MeanMetric(),
                        "x_loss t=[75,100)": MeanMetric(),
                        "t_avg": MeanMetric(),
                        "valid_rate": MeanMetric(),
                        "unique_rate": MeanMetric(),
                        "novel_rate": MeanMetric(),
                        "mol_pred_loaded": MeanMetric(),
                        "sanitization": MeanMetric(),
                        "inchi_convertible": MeanMetric(),
                        "all_atoms_connected": MeanMetric(),
                        "bond_lengths": MeanMetric(),
                        "bond_angles": MeanMetric(),
                        "internal_steric_clash": MeanMetric(),
                        "aromatic_ring_flatness": MeanMetric(),
                        "double_bond_flatness": MeanMetric(),
                        "internal_energy": MeanMetric(),
                        "sampling_time": MeanMetric(),
                    }
                ),
                "qmof150": ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "x_loss": MeanMetric(),
                        "x_loss t=[0,25)": MeanMetric(),
                        "x_loss t=[25,50)": MeanMetric(),
                        "x_loss t=[50,75)": MeanMetric(),
                        "x_loss t=[75,100)": MeanMetric(),
                        "t_avg": MeanMetric(),
                        "valid_rate": MeanMetric(),
                        "unique_rate": MeanMetric(),
                        "has_carbon": MeanMetric(),
                        "has_hydrogen": MeanMetric(),
                        "has_atomic_overlaps": MeanMetric(),
                        "has_overcoordinated_c": MeanMetric(),
                        "has_overcoordinated_n": MeanMetric(),
                        "has_overcoordinated_h": MeanMetric(),
                        "has_undercoordinated_c": MeanMetric(),
                        "has_undercoordinated_n": MeanMetric(),
                        "has_undercoordinated_rare_earth": MeanMetric(),
                        "has_metal": MeanMetric(),
                        "has_lone_molecule": MeanMetric(),
                        "has_high_charges": MeanMetric(),
                        # "is_porous": MeanMetric(),
                        "has_suspicicious_terminal_oxo": MeanMetric(),
                        "has_undercoordinated_alkali_alkaline": MeanMetric(),
                        "has_geometrically_exposed_metal": MeanMetric(),
                        # 'has_3d_connected_graph': MeanMetric(),
                        "all_checks": MeanMetric(),
                        "sampling_time": MeanMetric(),
                    }
                ),
            }
        )
        self.test_metrics = copy.deepcopy(self.val_metrics)

        # load bincounts for sampling
        self.num_nodes_bincount = {
            "mp20": torch.nn.Parameter(
                torch.load(
                    os.path.join(self.hparams.sampling.data_dir, f"mp_20/num_nodes_bincount.pt"),
                    map_location="cpu",
                ),
                requires_grad=False,
            ),
            "qm9": torch.nn.Parameter(
                torch.load(
                    os.path.join(self.hparams.sampling.data_dir, f"qm9/num_nodes_bincount.pt"),
                    map_location="cpu",
                ),
                requires_grad=False,
            ),
            "qmof150": torch.nn.Parameter(
                torch.load(
                    os.path.join(self.hparams.sampling.data_dir, f"qmof/num_nodes_bincount.pt"),
                    map_location="cpu",
                ),
                requires_grad=False,
            ),
        }
        self.spacegroups_bincount = {
            "mp20": torch.nn.Parameter(
                torch.load(
                    os.path.join(self.hparams.sampling.data_dir, f"mp_20/spacegroups_bincount.pt"),
                    map_location="cpu",
                ),
                requires_grad=False,
            ),
            "qm9": None,
            "qmof150": None,
        }

    def forward(self, batch: Data, sample_posterior: bool = True):
        # Encode batch to latent space
        with torch.no_grad():
            encoded_batch = self.autoencoder.encode(batch)
            # if sample_posterior:
            #     encoded_batch["x"] = encoded_batch["posterior"].sample()
            # else:
            #     encoded_batch["x"] = encoded_batch["posterior"].mode()
            # x_1 = encoded_batch["x"]

            # # Convert from PyG batch to dense batch with padding
            # x_1, mask = to_dense_batch(x_1, encoded_batch["batch"])

            ##### modified by RZD
            # sample continuous posterior (z_cont) as before
            if sample_posterior:
                z_cont = encoded_batch["posterior"].sample()
            else:
                z_cont = encoded_batch["posterior"].mode()
            # get deterministic type channel (z_type)
            # Prefer an explicit z_type produced by the VAE encode(); if absent, build from batch.atom_types
            if "z_type" in encoded_batch:
                z_type = encoded_batch["z_type"]
            else:
                # fallback: derive from batch.atom_types (ensures backward compatibility)
                z_type = self.autoencoder.atom_types_to_ztype(batch.atom_types).to(z_cont.dtype)

            # ensure z_type is on same device and has same ordering
            z_type = z_type.to(z_cont.device)

            # concat to form full latent (total_nodes, d+1)
            encoded_batch["x"] = torch.cat([z_type, z_cont], dim=-1)

            # for clarity keep x_cont separately too
            encoded_batch["x_cont"] = z_cont
            encoded_batch["z_type"] = z_type

            # x_1 is the full latent we will densify
            x_1 = encoded_batch["x"]

            # Convert from PyG batch to dense batch with padding
            x_1, mask = to_dense_batch(x_1, encoded_batch["batch"])
            # --- START MODIFICATION ---
            # batch.atom_types contains the atomic numbers (e.g., 1 for H, 6 for C)
            # We must "densify" it to shape [Batch, Max_Nodes] to match x_1
            atom_types_dense, _ = to_dense_batch(batch.atom_types, encoded_batch["batch"])
            # Add atom_types to the dictionary passed to the interpolant
            dense_encoded_batch = {
                "x_1": x_1,
                "token_mask": mask,
                "diffuse_mask": mask,
                "atom_types": atom_types_dense,  # <--- NEW FIELD
            }
            # --- END MODIFICATION ---

            # dense_encoded_batch = {"x_1": x_1, "token_mask": mask, "diffuse_mask": mask}

        # Corrupt batch using the interpolant
        self.interpolant.device = dense_encoded_batch["x_1"].device
        noisy_dense_encoded_batch = self.interpolant.corrupt_batch(dense_encoded_batch)

        ### modified by RZD
        # --- START: Ensure first channel (type channel) is not corrupted ---
        # dense_encoded_batch["atom_types"] has shape (B, N) of integer atomic numbers
        # noisy_dense_encoded_batch["x_t"] has shape (B, N, d_total)
        if "atom_types" in dense_encoded_batch and "x_t" in noisy_dense_encoded_batch:
            # build z_type tensor to match the latent type encoding used in the VAE
            # NOTE: in vae_module we encoded types as float atom ids in channel 0
            z_type = dense_encoded_batch["atom_types"].to(
                noisy_dense_encoded_batch["x_t"].dtype
            ).unsqueeze(-1)  # (B, N, 1)

            # put on same device
            z_type = z_type.to(noisy_dense_encoded_batch["x_t"].device)

            # replace the first channel of the noised latent with the deterministic type channel
            noisy_dense_encoded_batch["x_t"][..., 0:1] = z_type

            # Also ensure gt (x_1) first channel remains the deterministic type channel (sanity)
            noisy_dense_encoded_batch["x_1"][..., 0:1] = dense_encoded_batch["x_1"][..., 0:1]
        # --- END: Ensure first channel (type channel) is not corrupted ---
        
        # ---------------- DEBUG SANITY CHECK (forward) ----------------
        # Print small diagnostic showing that the first channel (type) is fixed
        if getattr(self, "global_rank", 0) == 0 and DEBUG_LDM_LATENT:
            try:
                # print first batch element's first 20 node types (dense)
                device = noisy_dense_encoded_batch["x_t"].device
                x_t_first = noisy_dense_encoded_batch["x_t"][0, :20, 0].detach().cpu().tolist()
                atom_types_first = dense_encoded_batch["atom_types"][0, :20].detach().cpu().tolist()
                x1_first = noisy_dense_encoded_batch["x_1"][0, :20, 0].detach().cpu().tolist()
                print(f"[DEBUG forward] x_t first-channel (first 20): {x_t_first}")
                print(f"[DEBUG forward] atom_types_dense (first 20):    {atom_types_first}")
                print(f"[DEBUG forward] x_1   first-channel (first 20): {x1_first}")
            except Exception as e:
                print(f"[DEBUG forward] sanity-check failed: {e}")
        # ---------------- END DEBUG ----------------


        # Prepare conditioning inputs to forward pass
        dataset_idx = batch.dataset_idx + 1  # 0 -> null class
        # if not self.hparams.conditioning.dataset_idx:
        #     dataset_idx = torch.zeros_like(dataset_idx)
        spacegroup = batch.spacegroup
        if not self.hparams.conditioning.spacegroup:
            spacegroup = torch.zeros_like(batch.spacegroup)

        # Use self-conditioning for ~half training batches
        if (
            self.interpolant.self_condition
            and random.random() < self.interpolant.self_condition_prob
        ):
            with torch.no_grad():
                x_sc = self.denoiser(
                    x=noisy_dense_encoded_batch["x_t"],
                    t=noisy_dense_encoded_batch["t"],
                    dataset_idx=dataset_idx,
                    spacegroup=spacegroup,
                    mask=mask,
                    x_sc=None,
                )
        else:
            x_sc = None

        # Run denoiser model
        pred_x = self.denoiser(
            x=noisy_dense_encoded_batch["x_t"],
            t=noisy_dense_encoded_batch["t"],
            dataset_idx=dataset_idx,
            spacegroup=spacegroup,
            mask=mask,
            x_sc=x_sc,
        )

        return pred_x, noisy_dense_encoded_batch

    def criterion(
        self,
        noisy_dense_encoded_batch: Dict[str, torch.Tensor],
        pred_x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Compute MSE loss w/ masking for padded tokens
        # gt_x_1 = noisy_dense_encoded_batch["x_1"]
        # norm_scale = 1 - torch.min(noisy_dense_encoded_batch["t"].unsqueeze(-1), torch.tensor(0.9))
        # x_error = (gt_x_1 - pred_x) / norm_scale
        # loss_mask = (
        #     noisy_dense_encoded_batch["token_mask"] * noisy_dense_encoded_batch["diffuse_mask"]
        # )
        # loss_denom = torch.sum(loss_mask, dim=-1) * pred_x.size(-1)
        # x_loss = torch.sum(x_error**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        # loss_dict = {"loss": x_loss.mean(), "x_loss": x_loss}

        ### modified by RZD
        # Compute MSE loss w/ masking for padded tokens
        gt_x_1 = noisy_dense_encoded_batch["x_1"]
        norm_scale = 1 - torch.min(noisy_dense_encoded_batch["t"].unsqueeze(-1), torch.tensor(0.9))
        # full per-channel error
        x_error = (gt_x_1 - pred_x) / norm_scale  # shape (B, N, D_total)
        # ZERO OUT error for the first (type) channel so loss only acts on continuous channels
        # (This makes loss focus on channels 1: rather than channel 0)
        x_error[..., 0] = 0.0
        loss_mask = (
            noisy_dense_encoded_batch["token_mask"] * noisy_dense_encoded_batch["diffuse_mask"]
        )
        # Adjust denominator: we have one fewer active channels (D_total - 1)
        num_active_channels = pred_x.size(-1) - 1
        loss_denom = torch.sum(loss_mask, dim=-1) * max(num_active_channels, 1)
        # compute MSE over channels excluding the first (because its error is zeroed)
        x_loss = torch.sum(x_error**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        loss_dict = {"loss": x_loss.mean(), "x_loss": x_loss}

        # ---------------- DEBUG SANITY CHECK (criterion) ----------------
        # Confirm that the error on the first channel is zeroed and that the denominator accounts for D-1 channels
        if getattr(self, "global_rank", 0) == 0 and DEBUG_LDM_LATENT:
            try:
                # sample first batch element
                err0 = x_error[0, :, 0].detach().cpu()
                nonzero_err0 = (err0.abs() > 1e-8).sum().item()
                print(f"[DEBUG criterion] non-zero elements in x_error[...,0] (should be 0): {nonzero_err0}")
                print(f"[DEBUG criterion] pred_x.shape: {pred_x.shape}, gt_x_1.shape: {gt_x_1.shape}")
                print(f"[DEBUG criterion] loss_mask sum (tokens): {loss_mask[0].sum().item()}")
                print(f"[DEBUG criterion] num_active_channels (D-1): {pred_x.size(-1)-1}")
            except Exception as e:
                print(f"[DEBUG criterion] sanity-check failed: {e}")
        # ---------------- END DEBUG ----------------


        # add diffusion loss stratified across t
        num_bins = 4
        flat_losses = x_loss.detach().cpu().numpy().flatten()
        flat_t = noisy_dense_encoded_batch["t"].detach().cpu().numpy().flatten()
        bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
        bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
        t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
        t_binned_n = np.bincount(bin_idx)
        for t_bin in np.unique(bin_idx).tolist():
            bin_start = bin_edges[t_bin]
            bin_end = bin_edges[t_bin + 1]
            t_range = f"x_loss t=[{int(bin_start*100)},{int(bin_end*100)})"
            range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
            loss_dict[t_range] = range_loss
        loss_dict["t_avg"] = np.mean(flat_t)

        return loss_dict

    #####################################################################################################

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for dataset in self.val_metrics.keys():
            for metric in self.val_metrics[dataset].values():
                metric.reset()

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when a training epoch starts."""
        for metric in self.train_metrics.values():
            metric.reset()

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        with torch.no_grad():
            # save masks used to apply augmentations
            sample_is_periodic = batch.dataset_idx != DATASET_TO_IDX["qm9"]
            node_is_periodic = sample_is_periodic[batch.batch]

            if self.hparams.augmentations.frac_coords == True:
                if node_is_periodic.any():
                    # sample random translation vector from batch length distribution / 2
                    random_translation = (
                        torch.normal(
                            torch.abs(batch.lengths.mean(dim=0)),
                            torch.abs(batch.lengths.std(dim=0).nan_to_num(1e-8)),
                        )
                        / 2
                    )
                    # apply same random translation to all Cartesian coordinates
                    pos_aug = batch.pos + random_translation
                    batch.pos = pos_aug
                    # compute new fractional coordinates for samples which are periodic
                    cell_per_node_inv = torch.linalg.inv(batch.cell[batch.batch][node_is_periodic])
                    frac_coords_aug = torch.einsum(
                        "bi,bij->bj", batch.pos[node_is_periodic], cell_per_node_inv
                    )
                    frac_coords_aug = frac_coords_aug % 1.0
                    batch.frac_coords[node_is_periodic] = frac_coords_aug

            if self.hparams.augmentations.pos == True:
                rot_mat = random_rotation_matrix(validate=True, device=self.device)
                pos_aug = batch.pos @ rot_mat.T
                batch.pos = pos_aug
                cell_aug = batch.cell @ rot_mat.T
                batch.cell = cell_aug
                # fractional coordinates are rotation invariant
                # assert torch.allclose(
                #     batch.frac_coords,
                #     torch.einsum("bi,bij->bj", pos_aug, torch.linalg.inv(cell_aug)[batch.batch]) % 1.0,
                #     rtol=1e-3,
                #     atol=1e-3,
                # )

        # forward pass
        pred_x, noisy_dense_encoded_batch = self.forward(batch)

        # calculate loss
        loss_dict = self.criterion(noisy_dense_encoded_batch, pred_x)

        # log relative proportions of datasets in batch
        loss_dict["dataset_idx"] = batch.dataset_idx.detach().flatten()

        # update and log train metrics
        for k, v in loss_dict.items():
            self.train_metrics[k](v)
            self.log(
                f"train/{k}",
                self.train_metrics[k],
                on_step=True,
                on_epoch=False,
                prog_bar=False if k != "loss" else True,
            )

        # return loss or backpropagation will fail
        return loss_dict["loss"]

    #####################################################################################################

    def on_validation_epoch_start(self) -> None:
        self.on_evaluation_epoch_start(stage="val")

    def validation_step(self, batch: Data, batch_idx: int, dataloader_idx: int) -> None:
        self.evaluation_step(batch, batch_idx, dataloader_idx, stage="val")

    def on_validation_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(stage="val")

    #####################################################################################################

    def on_test_epoch_start(self) -> None:
        self.on_evaluation_epoch_start(stage="test")

    def test_step(self, batch: Data, batch_idx: int, dataloader_idx: int) -> None:
        self.evaluation_step(batch, batch_idx, dataloader_idx, stage="test")

    def on_test_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(stage="test")

    #####################################################################################################

    def on_evaluation_epoch_start(self, stage: Literal["val", "test"]) -> None:
        "Lightning hook that is called when a validation/test epoch starts."
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        for dataset in metrics.keys():
            for metric in metrics[dataset].values():
                metric.reset()
        generation_evaluators = getattr(self, f"{stage}_generation_evaluators")
        for dataset in generation_evaluators.keys():
            generation_evaluators[dataset].clear()  # clear lists for next epoch

    def evaluation_step(
        self,
        batch: Data,
        batch_idx: int,
        dataloader_idx: int,
        stage: Literal["val", "test"],
    ) -> None:
        """Perform a single evaluation step on a batch of data from the validation/test set."""

        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")[IDX_TO_DATASET[dataloader_idx]]
        generation_evaluator = getattr(self, f"{stage}_generation_evaluators")[
            IDX_TO_DATASET[dataloader_idx]
        ]
        generation_evaluator.device = metrics["loss"].device

        # forward pass
        pred_x, noisy_dense_encoded_batch = self.forward(batch)

        # calculate loss
        loss_dict = self.criterion(noisy_dense_encoded_batch, pred_x)

        # update and log per-step val metrics
        for k, v in loss_dict.items():
            metrics[k](v)
            self.log(
                f"{stage}_{IDX_TO_DATASET[dataloader_idx]}/{k}",
                metrics[k],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                add_dataloader_idx=False,
            )

    def on_evaluation_epoch_end(self, stage: Literal["val", "test"]) -> None:
        """Lightning hook that is called when a validation/test epoch ends."""

        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        generation_evaluators = getattr(self, f"{stage}_generation_evaluators")

        for dataset in metrics.keys():
            if dataset not in self.val_generation_evaluators:  ### Added by RZD
                continue
            generation_evaluators[dataset].device = metrics[dataset]["loss"].device
            # --- MODIFIED: Get the specific dataloader for this dataset  RZD---
            # We need real batches (atom types) for CSP, so we grab the loader.
            # Note: self.trainer.datamodule.val_dataloader() returns a list of loaders.
            dataset_idx_int = DATASET_TO_IDX[dataset]
            if stage == "val":
                # Access the list of validation dataloaders
                # Depending on your lightning version/setup, this might be accessed via trainer
                # or directly from the datamodule attached to the trainer.
                dataloader = self.trainer.val_dataloaders[dataset_idx_int]
                # Alternative safety fallback if the above is strictly a list of 1 in some configs:
                # dataloader = self.trainer.datamodule.val_dataloader()[dataset_idx_int]
            else:
                dataloader = self.trainer.test_dataloaders[dataset_idx_int]

            t_start = time.time()
            samples_so_far = 0

            # --- MODIFIED: Iterate over the Dataloader instead of range() ---
            for batch in tqdm(dataloader, desc=f"    CSP Sampling {dataset}"):
                # Stop if we have generated enough samples
                if samples_so_far >= self.hparams.sampling.num_samples:
                    break

                batch = batch.to(self.device)
                # Perform CSP Sampling using the real batch
                # Pass only the arguments defined in your new sample_and_decode_csp function
                out, batch, samples = self.sample_and_decode_csp(
                    batch=batch,
                    cfg_scale=self.hparams.sampling.cfg_scale,
                )

                # for dataset in metrics.keys():
                #     generation_evaluators[dataset].device = metrics[dataset]["loss"].device
                #     t_start = time.time()
                #     for samples_so_far in tqdm(
                #         range(0, self.hparams.sampling.num_samples, self.hparams.sampling.batch_size),
                #         desc=f"    Sampling",
                #     ):
                # Perform sampling and decoding to crystal structures
                # out, batch, samples = self.sample_and_decode(
                #     num_nodes_bincount=self.num_nodes_bincount[dataset],
                #     spacegroups_bincount=self.spacegroups_bincount[dataset],
                #     batch_size=self.hparams.sampling.batch_size,
                #     cfg_scale=self.hparams.sampling.cfg_scale,
                #     dataset_idx=DATASET_TO_IDX[dataset],
                # )

                # Save predictions for metrics and visualisation
                start_idx = 0
                for idx_in_batch, num_atom in enumerate(batch["num_atoms"].tolist()):
                    _atom_types = (
                        out["atom_types"].narrow(0, start_idx, num_atom).argmax(dim=1)
                    )  # take argmax
                    _atom_types[_atom_types == 0] = 1  # atom type 0 -> 1 (H) to prevent crash
                    _pos = out["pos"].narrow(0, start_idx, num_atom) * 10.0  # nm to A
                    _frac_coords = out["frac_coords"].narrow(0, start_idx, num_atom)
                    _lengths = out["lengths"][idx_in_batch] * float(num_atom) ** (
                        1 / 3
                    )  # unscale lengths
                    _angles = torch.rad2deg(out["angles"][idx_in_batch])  # convert to degrees
                    generation_evaluators[dataset].append_pred_array(
                        {
                            "atom_types": _atom_types.detach().cpu().numpy(),
                            "pos": _pos.detach().cpu().numpy(),
                            "frac_coords": _frac_coords.detach().cpu().numpy(),
                            "lengths": _lengths.detach().cpu().numpy(),
                            "angles": _angles.detach().cpu().numpy(),
                            "sample_idx": samples_so_far
                            + self.global_rank * len(batch["num_atoms"])
                            + idx_in_batch,
                        }
                    )
                    start_idx = start_idx + num_atom
            t_end = time.time()

            # Compute generation metrics
            ### modified by RZD
            if len(generation_evaluators[dataset].pred_arrays_list) == 0:
                log.warning(f"[{stage}] No predictions for dataset={dataset}, skipping metrics.")
                continue
            ### END
            gen_metrics_dict = generation_evaluators[dataset].get_metrics(
                save=self.hparams.sampling.visualize,
                save_dir=self.hparams.sampling.save_dir + f"/{dataset}_{stage}_{self.global_rank}",
            )
            gen_metrics_dict["sampling_time"] = t_end - t_start
            for k, v in gen_metrics_dict.items():
                metrics[dataset][k](v)
                self.log(
                    f"{stage}_{dataset}/{k}",
                    metrics[dataset][k],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False if k != "valid_rate" else True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

            if self.hparams.sampling.visualize and type(self.logger) == WandbLogger:
                pred_table = generation_evaluators[dataset].get_wandb_table(
                    current_epoch=self.current_epoch,
                    save_dir=self.hparams.sampling.save_dir
                    + f"/{dataset}_{stage}_{self.global_rank}",
                )
                self.logger.experiment.log(
                    {f"{dataset}_{stage}_samples_table_device{self.global_rank}": pred_table}
                )

    #####################################################################################################

    def sample_and_decode(
        self,
        num_nodes_bincount,
        spacegroups_bincount,
        batch_size,
        cfg_scale=4.0,
        dataset_idx=0,
    ):
        # sample random lengths from distribution: (B, 1)
        sample_lengths = torch.multinomial(
            num_nodes_bincount.float(),
            batch_size,
            replacement=True,
        ).to(self.device)

        # create dataset_idx tensor
        # NOTE 0 -> null class within DiT, while 0 -> MP20 elsewhere, so increment by 1
        dataset_idx = torch.full(
            (batch_size,), dataset_idx + 1, dtype=torch.int64, device=self.device
        )

        # create spacegroup tensor
        if not self.hparams.conditioning.spacegroup or spacegroups_bincount is None:
            # null spacegroup
            spacegroup = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        else:
            # sample random spacegroups from distribution: (B, 1)
            spacegroup = torch.multinomial(
                spacegroups_bincount.float(),
                batch_size,
                replacement=True,
            ).to(self.device)

        # create token mask for visualization
        token_mask = torch.zeros(
            batch_size,
            max(sample_lengths),
            dtype=torch.bool,
            device=self.device,
        )
        for idx, length in enumerate(sample_lengths):
            token_mask[idx, :length] = True

        # create new samples from interpolant
        samples = self.interpolant.sample_with_classifier_free_guidance(
            batch_size=batch_size,
            # num_tokens=max(sample_lengths),
            num_tokens = int(sample_lengths), ### modified by RZD
            emb_dim=self.denoiser.d_x,
            model=self.denoiser,
            dataset_idx=dataset_idx,
            spacegroup=spacegroup,
            cfg_scale=cfg_scale,
            token_mask=token_mask,
        )
        # get final samples and remove padding (to PyG format)
        x = samples["clean_traj"][-1][token_mask]

        batch = {
            "x": x,
            "num_atoms": sample_lengths,
            "batch": torch.repeat_interleave(
                torch.arange(len(sample_lengths), device=self.device), sample_lengths
            ),
            "token_idx": (torch.cumsum(token_mask, dim=-1, dtype=torch.int64) - 1)[token_mask],
        }
        # decode samples to crystal structures using frozen decoder
        out = self.autoencoder.decode(batch)
        return out, batch, samples

    def sample_and_decode_csp(self, batch, cfg_scale=4.0):
        """CSP Sampling using a real batch from the dataset."""

        # 1. Extract Info from Batch
        batch_size = batch.num_graphs
        # Convert atom_types to dense [B, N]
        atom_types_dense, token_mask = to_dense_batch(batch.atom_types, batch.batch)
        sample_lengths = atom_types_dense.size(1)

        # IMPORTANT: real per-graph number of atoms (B,)
        num_atoms = batch.num_atoms  # tensor shape [B] ### modified by RZD
        # 2. Prepare Inputs
        dataset_idx = batch.dataset_idx + 1
        spacegroup = batch.spacegroup

        # 3. Call Interpolant with atom_types
        samples = self.interpolant.sample_with_classifier_free_guidance(
            batch_size=batch_size,
            # num_tokens=max(sample_lengths),  ## N=20 for MP_20
            num_tokens = int(sample_lengths), ### modified by RZD
            emb_dim=self.denoiser.d_x,
            model=self.denoiser,
            dataset_idx=dataset_idx,
            spacegroup=spacegroup,
            cfg_scale=cfg_scale,
            token_mask=token_mask,
            atom_types=atom_types_dense,  #
        )

        # get final samples and remove padding (to PyG format)
        x = samples["clean_traj"][-1][token_mask]

        batch = {
            "x": x,
            # "num_atoms": sample_lengths,
            "num_atoms": num_atoms,  # <-- MUST be (B,) ### RZD
            "batch": torch.repeat_interleave(
                # torch.arange(len(sample_lengths), device=self.device), sample_lengths
                # torch.arange(sample_lengths, device=self.device), sample_lengths ### has error
                torch.arange(batch_size, device=self.device), num_atoms ### modified by RZD
            ),
            "token_idx": (torch.cumsum(token_mask, dim=-1, dtype=torch.int64) - 1)[token_mask],
        }
        if  DEBUG_LDM_LATENT:
            print(f"[DEBUG CSP] x: {x.shape}, batch: {batch['batch'].shape} sum(num_atoms): {int(num_atoms.sum())}")
        # decode samples to crystal structures using frozen decoder
        out = self.autoencoder.decode(batch)
        return out, batch, samples

    #####################################################################################################

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        try:
            # Clear cache for Equiformer SO3 embeddings
            self.autoencoder.encoder.mappingReduced.device = self.device
            self.autoencoder.encoder.mappingReduced.mask_indices_cache = None
            self.autoencoder.encoder.mappingReduced.rotate_inv_rescale_cache = None
            for rotation_module in self.autoencoder.encoder.SO3_rotation:
                rotation_module.mapping.device = self.device
                rotation_module.mapping.mask_indices_cache = None
                rotation_module.mapping.rotate_inv_rescale_cache = None
            log.info("Clear Equiformer checkpoint SO3 rotation mapping cache.")
        except AttributeError:
            pass

        if self.hparams.compile and stage == "fit":
            self.autoencoder = torch.compile(self.autoencoder)
            self.denoiser = torch.compile(self.denoiser)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_mp20/valid_rate",
                    "interval": "epoch",
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}
