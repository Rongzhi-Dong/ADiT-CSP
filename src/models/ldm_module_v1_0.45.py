"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import copy
import os
import random
import time
from typing import Any, Dict, Literal, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch.nn import ModuleDict
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torchmetrics import MeanMetric
from tqdm import tqdm

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
        self.val_generation_evaluators = {
            "mp20": CrystalGenerationEvaluator(
                dataset_cif_list=pd.read_csv(
                    os.path.join(self.hparams.sampling.data_dir, f"mp_20/raw/all.csv")
                )["cif"].tolist()
            ),
            "qm9": MoleculeGenerationEvaluator(
                dataset_smiles_list=torch.load(
                    os.path.join(self.hparams.sampling.data_dir, f"qm9/smiles.pt"),
                ),
                removeHs=self.hparams.sampling.removeHs,
            ),
            "qmof150": MOFGenerationEvaluator(),
        }
        self.test_generation_evaluators = copy.deepcopy(self.val_generation_evaluators)

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
        # with torch.no_grad():
        #     encoded_batch = self.autoencoder.encode(batch)
        #     if sample_posterior:
        #         encoded_batch["x"] = encoded_batch["posterior"].sample()
        #     else:
        #         encoded_batch["x"] = encoded_batch["posterior"].mode()
        #     x_1 = encoded_batch["x"]

        #     # Convert from PyG batch to dense batch with padding
        #     x_1, mask = to_dense_batch(x_1, encoded_batch["batch"])
        #     dense_encoded_batch = {"x_1": x_1, "token_mask": mask, "diffuse_mask": mask}
        
        
        ######### modified by RZD  ####################
        # Encode batch to latent space
        with torch.no_grad():
            encoded_batch = self.autoencoder.encode(batch)
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
        ######### End modification ####################

        # Corrupt batch using the interpolant
        self.interpolant.device = dense_encoded_batch["x_1"].device
        noisy_dense_encoded_batch = self.interpolant.corrupt_batch(dense_encoded_batch)



        ######### modified by RZD  ####################
        # dense_encoded_batch["atom_types"] has shape (B, N) of integer atomic numbers
        # noisy_dense_encoded_batch["x_t"] has shape (B, N, d_total)
        if "atom_types" in dense_encoded_batch and "x_t" in noisy_dense_encoded_batch:
            # build z_type tensor to match the latent type encoding used in the VAE
            # NOTE: in vae_module we encoded types as float atom ids in channel 0
            # z_type = dense_encoded_batch["atom_types"].to(
            #     noisy_dense_encoded_batch["x_t"].dtype
            # ).unsqueeze(-1)  # (B, N, 1)

            # # put on same device
            # z_type = z_type.to(noisy_dense_encoded_batch["x_t"].device)

            ####### Feb10 ##########
            # FIX: convert integer atom types into the VAE's z_type representation
            # Use the autoencoder helper so the type-channel scale matches the VAE's latent space.
            z_type = self.autoencoder.atom_types_to_ztype(dense_encoded_batch["atom_types"].to(noisy_dense_encoded_batch["x_t"].device))
            # ensure dtype matches the latent tensor
            z_type = z_type.to(dtype=noisy_dense_encoded_batch["x_t"].dtype)

            # ensure contiguous memory before in-place assignment (avoids odd stride issues)
            z_type = z_type.contiguous()
            ##### END ##########



            # replace the first channel of the noised latent with the deterministic type channel
            noisy_dense_encoded_batch["x_t"][..., 0:1] = z_type

            # Also ensure gt (x_1) first channel remains the deterministic type channel (sanity)
            noisy_dense_encoded_batch["x_1"][..., 0:1] = dense_encoded_batch["x_1"][..., 0:1]
        ######### End modification ####################

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
        # # Compute MSE loss w/ masking for padded tokens
        # gt_x_1 = noisy_dense_encoded_batch["x_1"]
        # norm_scale = 1 - torch.min(noisy_dense_encoded_batch["t"].unsqueeze(-1), torch.tensor(0.9))
        # x_error = (gt_x_1 - pred_x) / norm_scale
        # loss_mask = (
        #     noisy_dense_encoded_batch["token_mask"] * noisy_dense_encoded_batch["diffuse_mask"]
        # )
        # loss_denom = torch.sum(loss_mask, dim=-1) * pred_x.size(-1)
        # x_loss = torch.sum(x_error**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        # loss_dict = {"loss": x_loss.mean(), "x_loss": x_loss}

        ######### modified by RZD  ####################
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
        ######### End modification ####################
        
        
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

    def build_lattice_matrix(self, lengths, angles):
        """
        Batch-compatible lattice matrix construction (Standard convention).
        lengths: (B, 3), angles: (B, 3) in radians
        """
        device = lengths.device
        a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]
        alpha, beta, gamma = angles[:, 0], angles[:, 1], angles[:, 2]

        # Batch initialization of components
        zeros = torch.zeros_like(a)
        
        # Vector a along x-axis
        ax, ay, az = a, zeros, zeros
        
        # Vector b in xy-plane
        bx = b * torch.cos(gamma)
        by = b * torch.sin(gamma)
        bz = zeros
        
        # Vector c
        cx = c * torch.cos(beta)
        cy = c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / torch.sin(gamma)
        # Clamp to avoid NaNs in sqrt
        cz = torch.sqrt(torch.clamp(c**2 - cx**2 - cy**2, min=1e-8))
        
        # Stack into (B, 3, 3) matrix
        # Each row is a lattice vector: [a], [b], [c]
        line1 = torch.stack([ax, ay, az], dim=1)
        line2 = torch.stack([bx, by, bz], dim=1)
        line3 = torch.stack([cx, cy, cz], dim=1)
        
        return torch.stack([line1, line2, line3], dim=1)


    def on_evaluation_epoch_end(self, stage: Literal["val", "test"]) -> None:
        """Lightning hook that is called when a validation/test epoch ends."""

        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        generation_evaluators = getattr(self, f"{stage}_generation_evaluators")

        for dataset in metrics.keys():
            generation_evaluators[dataset].device = metrics[dataset]["loss"].device
            
            ######### modified by RZD  ####################
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
            for batch_data in tqdm(dataloader, desc=f"CSP Sampling {dataset}"):
                # Stop if we have generated enough samples
                if samples_so_far >= self.hparams.sampling.num_samples:
                    break

                batch_data = batch_data.to(self.device)
                # Perform CSP Sampling using the real batch
                # Pass only the arguments defined in your new sample_and_decode_csp function
                out, decoder_batch, samples = self.sample_and_decode_csp(
                    batch=batch_data,
                    cfg_scale=self.hparams.sampling.cfg_scale,
                )
            ######### End modification  ####################
                        
            # t_start = time.time()
            # for samples_so_far in tqdm(
            #     range(0, self.hparams.sampling.num_samples, self.hparams.sampling.batch_size),
            #     desc=f"    Sampling",
            # ):
            #     # Perform sampling and decoding to crystal structures
            #     out, batch, samples = self.sample_and_decode(
            #         num_nodes_bincount=self.num_nodes_bincount[dataset],
            #         spacegroups_bincount=self.spacegroups_bincount[dataset],
            #         batch_size=self.hparams.sampling.batch_size,
            #         cfg_scale=self.hparams.sampling.cfg_scale,
            #         dataset_idx=DATASET_TO_IDX[dataset],
            #     )
                # Save predictions for metrics and visualisation
                start_idx = 0
                for idx_in_batch, num_atom in enumerate(decoder_batch["num_atoms"].tolist()):

                    _atom_types = (
                        out["atom_types"].narrow(0, start_idx, num_atom).argmax(dim=1)
                    )  # take argmax
                    
                    _atom_types[_atom_types == 0] = 1  # atom type 0 -> 1 (H) to prevent crash
                    # _pos = out["pos"].narrow(0, start_idx, num_atom) * 10.0  # nm to A
                    _pos_A = out["pos"].narrow(0, start_idx, num_atom) * 10.0  # Cartesian in Å ### RZD 
                    # _frac_coords = out["frac_coords"].narrow(0, start_idx, num_atom)% 1.0 ### RZD
                    _lengths = out["lengths"][idx_in_batch] * float(num_atom) ** (
                        1 / 3
                    )  # unscale lengths
                    _angles = torch.rad2deg(out["angles"][idx_in_batch])  # convert to degrees
                    cell = self.build_lattice_matrix(_lengths.unsqueeze(0), _angles.unsqueeze(0))[0]
                    cell_inv = torch.linalg.inv(cell)
                    _frac_coords_physical = (_pos_A @ cell_inv.T) 
                    _frac_coords_wrapped = _frac_coords_physical % 1.0
                    
                    # diagnostic comparison (print only for first structure in the first batch)
                    if idx_in_batch == 0 and samples_so_far == 0:
                        dec_frac = out["frac_coords"].narrow(0, start_idx, num_atom)
                        recomputed_frac = (_pos_A @ cell_inv.T) % 1.0
                        print("DEBUG_FRAC_COMPARE:")
                        print(" decoder frac (first 5):", dec_frac[:5].tolist())
                        print(" recomputed frac (first 5):", recomputed_frac[:5].tolist())
                        print(" recomputed frac span:", float(recomputed_frac.min()), float(recomputed_frac.max()))
                        print(" decoder frac span:", float(dec_frac.min()), float(dec_frac.max()))

                    generation_evaluators[dataset].append_pred_array(
                        {
                            "atom_types": _atom_types.detach().cpu().numpy(),
                            # "pos": _pos.detach().cpu().numpy(),
                            # "frac_coords": _frac_coords.detach().cpu().numpy(),
                            "pos": _pos_A.detach().cpu().numpy(),
                            "frac_coords": _frac_coords_wrapped.detach().cpu().numpy(),
                            "lengths": _lengths.detach().cpu().numpy(),
                            "angles": _angles.detach().cpu().numpy(),
                            "sample_idx": samples_so_far
                            + self.global_rank * len(decoder_batch["num_atoms"])
                            + idx_in_batch,
                        }
                    )
                    start_idx = start_idx + num_atom
                samples_so_far += batch_data.num_graphs
            t_end = time.time()

            ### modified by RZD #####
            if len(generation_evaluators[dataset].pred_arrays_list) == 0:
                log.warning(f"[{stage}] No predictions for dataset={dataset}, skipping metrics.")
                continue
            ### END modification ######

            # Compute generation metrics
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
            num_tokens=max(sample_lengths),
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

    '''
    def sample_and_decode_csp(self, batch, cfg_scale=4.0):
        """CSP Sampling using a real batch from the dataset."""
        ###  new function for csp by RZD   ######

        # 1. Extract Info from Batch
        batch_size = batch.num_graphs
        # Convert atom_types to dense [B, N]
        atom_types_dense, token_mask = to_dense_batch(batch.atom_types, batch.batch)
        sample_lengths = atom_types_dense.size(1)

        # IMPORTANT: real per-graph number of atoms (B,)
        num_atoms = batch.num_atoms  # tensor shape [B] ### modified by RZD
        
        
        # === DIAGNOSTIC 1: Input batch ===
        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC 1: Input Batch")
        print(f"{'='*60}")
        print(f"batch_size: {batch_size}")
        print(f"batch.num_atoms: {num_atoms}")
        print(f"batch.num_atoms sum: {num_atoms.sum()}")
        print(f"batch.batch unique counts: {torch.bincount(batch.batch)}")
        print(f"atom_types_dense shape: {atom_types_dense.shape}")
        print(f"token_mask shape: {token_mask.shape}")
        print(f"token_mask sum per sample: {token_mask.sum(dim=1)}")
        print(f"Match? {torch.all(token_mask.sum(dim=1) == num_atoms)}")        
            
        
        
        # 2. Prepare Inputs
        dataset_idx = batch.dataset_idx+1
        spacegroup = batch.spacegroup
        # print("*******", spacegroup)
        # exit()

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
        
        # === DIAGNOSTIC 2: Sampled latent ===
        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC 2: Sampled Latent")
        print(f"{'='*60}")
        print(f"x shape: {x.shape}")
        print(f"Expected total atoms: {token_mask.sum()}")
        print(f"Actual atoms in x: {x.shape[0]}")
        print(f"Match? {x.shape[0] == token_mask.sum()}")
        print(f"x channel 0 (types) range: [{x[..., 0].min():.2f}, {x[..., 0].max():.2f}]")
            
        #### newly added by RZD on Feb 9 ###############
        # # Verify type channel is correct (helpful for debugging)
        # if atom_types_dense is not None:
        #     atom_types_flat = atom_types_dense[token_mask]
        #     recovered_types = x[..., 0]
        #     type_error = (recovered_types - atom_types_flat).abs().max()
        #     if type_error > 0.5:
        #         log.warning(f"Type channel drift detected: max error = {type_error:.4f}")
        #     # Ensure exact types for decoding
        #     x[..., 0] = atom_types_flat.to(x.dtype)
        
        # CRITICAL: Set channel 0 to atom types (deterministic channel)
        # Extract atom types for non-padded positions
        # atom_types_flat = atom_types_dense[token_mask]  # (total_nodes,)
        # z_type = atom_types_flat.unsqueeze(-1).to(x.dtype)  # (total_nodes, 1)
        # # Replace channel 0 with deterministic type information
        # x = torch.cat([z_type, x[..., 1:]], dim=-1)  # Ensure channel 0 is types
        
        # atom_types_flat = batch.atom_types.to(x.dtype)
        # x = x.clone()
        # x[..., 0] = atom_types_flat
        x[..., 0] = self.autoencoder.atom_types_to_ztype(batch.atom_types).to(x.dtype)
        
        #### End #############


        # print(f"[DEBUG] Shapes:")
        # print(f"  atom_types_dense: {atom_types_dense.shape}")  # (B, max_N)
        # print(f"  token_mask: {token_mask.shape}")              # (B, max_N)
        # print(f"  num_atoms: {num_atoms.shape}")                # (B,)
        # print(f"  x (sampled): {x.shape}")                      # (sum(num_atoms), d+1)
        # print(f"  x channel 0 range: [{x[..., 0].min():.2f}, {x[..., 0].max():.2f}]")
        # print(f"  Expected atom types range: [{atom_types_dense.min()}, {atom_types_dense.max()}]")
        # # Verify type channel
        # atom_types_flat = atom_types_dense[token_mask]
        # type_diff = (x[..., 0] - atom_types_flat).abs()
        # print(f"  Type channel error: max={type_diff.max():.6f}, mean={type_diff.mean():.6f}")



        decoder_batch = {
            "x": x,
            # "num_atoms": sample_lengths,
            "num_atoms": num_atoms,  # <-- MUST be (B,) ### RZD
            "batch": torch.repeat_interleave(
                torch.arange(batch_size, device=self.device), num_atoms ### modified by RZD         
            ),
            # "token_idx": (torch.cumsum(token_mask, dim=-1, dtype=torch.int64) - 1)[token_mask],
            "token_idx": batch.token_idx if hasattr(batch, 'token_idx') else None,
        }
        # if  DEBUG_LDM_LATENT:
        #     print(f"[DEBUG CSP] x: {x.shape}, batch: {batch['batch'].shape} sum(num_atoms): {int(num_atoms.sum())}")
        # decode samples to crystal structures using frozen decoder
        
        # === DIAGNOSTIC 3: Decoder input ===
        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC 3: Decoder Input")
        print(f"{'='*60}")
        print(f"decoder_batch['x'] shape: {decoder_batch['x'].shape}")
        print(f"decoder_batch['num_atoms']: {decoder_batch['num_atoms']}")
        print(f"decoder_batch['batch'] unique: {torch.bincount(decoder_batch['batch'])}")
        print(f"decoder_batch['batch'] max: {decoder_batch['batch'].max()}")
        print(f"Expected batch indices: 0 to {batch_size - 1}")
        
        
        out = self.autoencoder.decode(decoder_batch)

        # === DIAGNOSTIC 4: Decoder output ===
        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC 4: Decoder Output")
        print(f"{'='*60}")
        print(f"out['atom_types'] shape: {out['atom_types'].shape}")
        print(f"out['pos'] shape: {out['pos'].shape}")
        print(f"out['lengths'] shape: {out['lengths'].shape}")
        print(f"out['angles'] shape: {out['angles'].shape}")
        print(f"Expected structures: {batch_size}")
        print(f"Actual structures in lengths: {out['lengths'].shape[0]}")
        print(f"{'='*60}\n")
        
        return out, decoder_batch, samples
    '''
    def sample_and_decode_csp(self, batch, cfg_scale=4.0):
        batch_size = batch.num_graphs
        num_atoms = batch.num_atoms
        
        # # Diagnostic 1: Check denoiser dimension
        # print(f"\n{'='*70}")
        # print(f"DIAGNOSTIC: Denoiser Configuration")
        # print(f"{'='*70}")
        # print(f"self.denoiser.d_x: {self.denoiser.d_x}")
        # print(f"emb_dim being used: {self.denoiser.d_x}")  # You have this currently
        # print(f"emb_dim should be: 9")  # What it should be
        
        # 1. Encode atom types
        with torch.no_grad():
            z_type_pyg = self.autoencoder.atom_types_to_ztype(batch.atom_types)
            z_type_dense, token_mask = to_dense_batch(z_type_pyg, batch.batch)
        
        # # Diagnostic 2: Check type encoding
        # print(f"\n{'='*70}")
        # print(f"DIAGNOSTIC: Type Encoding")
        # print(f"{'='*70}")
        # print(f"batch.atom_types (raw): {batch.atom_types[:10]}")
        # print(f"z_type_pyg (encoded): {z_type_pyg[:10]}")
        # print(f"z_type_dense shape: {z_type_dense.shape}")

        # 2. Sample
        samples = self.interpolant.sample_with_classifier_free_guidance(
            batch_size=batch_size,
            num_tokens=int(z_type_dense.size(1)),
            emb_dim=self.denoiser.d_x,  # Currently wrong
            model=self.denoiser,
            dataset_idx=batch.dataset_idx + 1,
            spacegroup=batch.spacegroup,
            cfg_scale=cfg_scale,
            token_mask=token_mask,
            atom_types=z_type_dense,
        )

        # 3. Extract and check
        x = samples["clean_traj"][-1][token_mask]
        
        ######## Feb 10 ########
        # Defensive: make contiguous and verify continuous channels are non-trivial
        x = x.contiguous()

        # Quick diagnostic to see if continuous latent collapsed
        cont = x[..., 1:]  # continuous channels
        cont_mean = cont.mean().item()
        cont_std = cont.std().item()


        # print(f"\n{'='*70}")
        # print(f"DECODER OUTPUT INVESTIGATION")
        # print(f"{'='*70}")
        # print(f"[DEBUG CSP] sampled latent shape: {x.shape}, cont mean={cont_mean:.6f}, cont std={cont_std:.6f}")
        # # optionally fail-fast to discover where it went wrong
        # if cont_std < 1e-6:
        #     print("⚠️  Continuous latent std is near zero — cont channels collapsed!")
        #     # Dump a small sample to inspect
        #     print("  sample cont (first row, first 12 dims):", cont[0, :12].cpu().tolist() if cont.numel() else "N/A")

        ######## END ###########


        # # Diagnostic 3: Check sampled latent
        # print(f"\n{'='*70}")
        # print(f"DIAGNOSTIC: Sampled Latent")
        # print(f"{'='*70}")
        # print(f"x.shape: {x.shape}")
        # print(f"x channel 0 (should be types): min={x[..., 0].min():.2f}, max={x[..., 0].max():.2f}")
        # print(f"Original atom types: min={batch.atom_types.min()}, max={batch.atom_types.max()}")
        
        # Compare channel 0 with expected types
        # atom_types_flat = batch.atom_types
        # if z_type_pyg.dim() == 1:
        #     z_type_pyg_compare = z_type_pyg
        # else:
        #     z_type_pyg_compare = z_type_pyg.squeeze(-1)
        
        # channel_0_values = x[..., 0]
        # diff = (channel_0_values - z_type_pyg_compare).abs()
        # # print(f"Difference between x[0] and z_type: max={diff.max():.6f}, mean={diff.mean():.6f}")
        
        # if diff.max() > 1.0:
        #     print(f"⚠️  WARNING: Type channel has DRIFTED significantly!")
        #     print(f"   This indicates type enforcement is NOT working!")
        
        # 4. Enforce type (fix shape first)
        if z_type_pyg.dim() == 1:
            z_type_pyg = z_type_pyg.unsqueeze(-1)
        x[..., 0:1] = z_type_pyg.to(x.dtype)
        
        # # Diagnostic 4: Check decoder input
        # print(f"\n{'='*70}")
        # print(f"DIAGNOSTIC: Decoder Input")
        # print(f"{'='*70}")
        # print(f"Decoder will receive x.shape: {x.shape}")
        

        # # ------------------- RESCALE sampled continuous channels (inference-time fix) -------------------
        # with torch.no_grad():
        #     # sampled continuous part
        #     sampled_cont = x[..., 1:]  # shape (total_nodes, d_cont)
        #     samp_mean = float(sampled_cont.mean().detach().cpu())
        #     samp_std = float(sampled_cont.std().detach().cpu().clamp(min=1e-8))

        #     # get encoder stats on the SAME real batch (what decoder expects)
        #     enc = self.autoencoder.encode(batch)
        #     # prefer posterior.mode if available (this matches your encode usage in forward)
        #     if "posterior" in enc and hasattr(enc["posterior"], "mode"):
        #         z_enc = enc["posterior"].mode()
        #     elif "posterior" in enc and hasattr(enc["posterior"], "sample"):
        #         z_enc = enc["posterior"].sample()
        #     elif "x" in enc:
        #         z_enc = enc["x"]
        #     else:
        #         # fallback: raise informative error
        #         raise RuntimeError("Could not extract encoder latent (expected enc['posterior'] or enc['x']).")

        #     enc_mean = float(z_enc.mean().detach().cpu())
        #     enc_std = float(z_enc.std().detach().cpu().clamp(min=1e-8))

        #     # Compute scaling factor
        #     scale = enc_std / samp_std
        #     shift = enc_mean - samp_mean * scale

        #     # Apply rescaling in-place (preserve dtype/device)
        #     # new = (old - samp_mean) * scale + enc_mean  <==> old*scale + shift
        #     x[..., 1:] = (x[..., 1:] - samp_mean) * scale + enc_mean

        #     # Debug prints (optional, remove or gate behind a debug flag)
        #     print(f"[SCALE FIX] sampled_cont mean/std = {samp_mean:.6f}/{samp_std:.6f}; encoder mean/std = {enc_mean:.6f}/{enc_std:.6f}")
        #     print(f"[SCALE FIX] applied scale={scale:.4f}, shift={shift:.6f}; new cont mean={float(x[...,1:].mean()):.6f}, new cont std={float(x[...,1:].std()):.6f}")
        # # -----------------------------------------------------------------------------------------------

        decoder_batch = {
            "x": x,
            "num_atoms": num_atoms,
            "batch": torch.repeat_interleave(
                torch.arange(batch_size, device=self.device), num_atoms
            ),
            "token_idx": batch.token_idx if hasattr(batch, 'token_idx') else None,
            "cell": batch.cell,
            "lengths_scaled": batch.lengths_scaled,
            "angles_radians": batch.angles_radians,
        }


        # # DIAG A: per-structure cont std + per-structure frac span after rescale
        # with torch.no_grad():
        #     cont = x[..., 1:]  # (total_nodes, d_cont)
        #     ptr = decoder_batch["batch"]  # (total_nodes,), values 0..B-1
        #     B = int(ptr.max().item()) + 1
        #     per_struct_stats = []
        #     start = 0
        #     print("\n[DIAG] per-structure cont std and resulting frac span (first 8 structures):")
        #     for i in range(min(B, 8)):
        #         # indices for this structure
        #         idxs = (ptr == i).nonzero(as_tuple=True)[0]
        #         if idxs.numel() == 0:
        #             continue
        #         cont_i = cont[idxs]  # (n_i, d_cont)
        #         std_per_dim = cont_i.std(dim=0).cpu().tolist()
        #         std_global = float(cont_i.std().cpu())
        #         # decode fractionals for this structure to compute span
        #         # build a one-structure decoder_batch and decode (cheap-ish)
        #         db = {"x": x[idxs].unsqueeze(0).reshape(-1, x.shape[-1]), "num_atoms": torch.tensor([idxs.numel()]), "batch": torch.zeros(idxs.numel(), dtype=torch.long)}
        #         # reuse existing decode by copying and adapting minimal fields (depends on your autoencoder.decode signature)
        #         # To avoid heavy decode here, we compute frac from pos returned earlier: we'll find pos slice by searching out.
        #         per_struct_stats.append((i, idxs.numel(), std_global, std_per_dim))
        #         print(f" struct {i}: n={idxs.numel():3d}, cont std global={std_global:.4f}, cont std per-dim (first 6)={std_per_dim[:6]}")
        # # DIAG B: PCA on cont latents for a representative structure (0)
        # # import torch
        # from torch import svd

        # with torch.no_grad():
        #     ptr = decoder_batch["batch"]
        #     idxs0 = (ptr == 0).nonzero(as_tuple=True)[0]
        #     if idxs0.numel() >= 4:
        #         cont0 = x[idxs0].float()[..., 1:]  # (n0, d_cont)
        #         # center
        #         cont0c = cont0 - cont0.mean(dim=0, keepdim=True)
        #         # SVD
        #         U, S, V = torch.svd(cont0c)  # S singular values
        #         sing = S.cpu().numpy()
        #         var_explained = (sing**2) / (sing**2).sum()
        #         print(f"[DIAG PCA] structure0 singular values (first 8): {sing[:8].tolist()}")
        #         print(f"[DIAG PCA] structure0 var explained (first 8): {var_explained[:8].tolist()}")
        #     else:
        #         print("[DIAG PCA] structure0 too small for PCA.")


        # ######## Feb 10 ######
        # # Assert shape & ordering
        # expected_latent_dim = self.autoencoder.latent_dim + 1 if hasattr(self.autoencoder, "latent_dim") else x.size(-1)
        # assert x.shape[-1] == expected_latent_dim, f"latent dim mismatch: got {x.shape[-1]}, expected {expected_latent_dim}"

        # # check type channel equals mapping (sanity)
        # expected_type = self.autoencoder.atom_types_to_ztype(batch.atom_types).to(x.dtype)
        # # check only first few entries to avoid heavy ops
        # if x.size(0) >= expected_type.size(0):
        #     if not torch.allclose(x[: expected_type.size(0), 0:1], expected_type, atol=1e-5):
        #         print("⚠️  type-channel mismatch before decode (sanity check failed).")


        # with torch.no_grad():
        #     # sampled cont (flat over all nodes)
        #     sampled_cont = x[..., 1:]  # shape (total_nodes, d_cont)
        #     print("[TEST] sampled cont mean/std per-dim:", sampled_cont.mean(dim=0).cpu().tolist(), sampled_cont.std(dim=0).cpu().tolist())
        #     print("[TEST] sampled cont global mean/std:", float(sampled_cont.mean()), float(sampled_cont.std()))

        #     # get encoder stats on the *same real batch* (what decoder expects)
        #     enc = self.autoencoder.encode(batch)
        #     # use posterior mode (or sample) as used during forward if applicable
        #     if "posterior" in enc:
        #         z_enc = enc["posterior"].mode() if hasattr(enc["posterior"], "mode") else enc["posterior"].sample()
        #     else:
        #         z_enc = enc["x"]  # fallback
        #     # z_enc is (total_nodes, d_cont) or (N, d_cont) matching order used earlier
        #     print("[TEST] encoder cont mean/std per-dim:", z_enc.mean(dim=0).cpu().tolist(), z_enc.std(dim=0).cpu().tolist())
        #     print("[TEST] encoder cont global mean/std:", float(z_enc.mean()), float(z_enc.std()))
        
        # with torch.no_grad():
        #     # compute scalars (global) — robust to zeros
        #     sampled_std = float(sampled_cont.std().clamp(min=1e-8))
        #     enc_std = float(z_enc.std().clamp(min=1e-8))
        #     sampled_mean = float(sampled_cont.mean())
        #     enc_mean = float(z_enc.mean())

        #     scale = enc_std / sampled_std
        #     print(f"[TEST] scaling sampled cont by factor {scale:.4f} (enc_std {enc_std:.4f}, samp_std {sampled_std:.4f})")

        #     x_test = x.clone()
        #     x_test[..., 1:] = (x_test[..., 1:] - sampled_mean) * scale + enc_mean

        #     # rebuild decoder_batch.x pointer to x_test (same decoder_batch in your code)
        #     decoder_batch_test = dict(decoder_batch)
        #     decoder_batch_test["x"] = x_test

        #     out_test = self.autoencoder.decode(decoder_batch_test)

        #     # print a quick frac check for first structure
        #     start_idx = 0
        #     for idx in range(min(3, batch_size)):
        #         n = num_atoms[idx].item()
        #         frac_test = out_test["frac_coords"][start_idx : start_idx + n]
        #         print(f"[TEST-DECODE] struct {idx} frac range after rescale: [{frac_test.min():.4f}, {frac_test.max():.4f}]")
        #         start_idx += n


        # with torch.no_grad():
        #     # take z_type from x (or compute from batch)
        #     z_type_flat = x[..., 0:1].clone()

        #     # make cont noise at encoder scale
        #     noise = torch.randn_like(x[..., 1:]) * float(z_enc.std())
        #     x_manual = torch.cat([z_type_flat, noise], dim=-1)

        #     decoder_batch_manual = dict(decoder_batch)
        #     decoder_batch_manual["x"] = x_manual

        #     out_manual = self.autoencoder.decode(decoder_batch_manual)
        #     # print first struct frac range
        #     start_idx = 0
        #     for idx in range(min(3, batch_size)):
        #         n = num_atoms[idx].item()
        #         frac_m = out_manual["frac_coords"][start_idx : start_idx + n]
        #         print(f"[MANUAL-DECODE] struct {idx} frac range with random cont noise: [{frac_m.min():.4f}, {frac_m.max():.4f}]")
        #         start_idx += n

        # ######## END #########    


        out = self.autoencoder.decode(decoder_batch)
    
        # # After: out = self.autoencoder.decode(decoder_batch)
        # print(f"\n{'='*70}")
        # print(f"DECODER OUTPUT INVESTIGATION")
        # print(f"{'='*70}")

        # # Test: Recompute fractional coords from positions manually
        # start_idx = 0
        # for idx in range(min(3, batch_size)):  # Check first 3 structures
        #     n = num_atoms[idx].item()
            
        #     # Get outputs
        #     pos = out["pos"][start_idx:start_idx + n]  # (n, 3) in nm
        #     frac_from_decoder = out["frac_coords"][start_idx:start_idx + n]  # (n, 3)
        #     lengths = out["lengths"][idx]  # (3,) scaled
        #     angles = out["angles"][idx]  # (3,) in radians
            
        #     print(f"\nStructure {idx} ({n} atoms):")
        #     print(f"  Positions (nm): range [{pos.min():.4f}, {pos.max():.4f}]")
        #     print(f"  Frac from decoder: range [{frac_from_decoder.min():.4f}, {frac_from_decoder.max():.4f}]")
        #     print(f"  Lengths (scaled): {lengths}")
        #     print(f"  Angles (rad): {angles}")
            
        #     # Manually recompute fractional coordinates
        #     # Step 1: Build cell matrix
        #     unscaled_lengths = lengths * (n ** (1/3))
        #     a, b, c = unscaled_lengths[0], unscaled_lengths[1], unscaled_lengths[2]
        #     alpha, beta, gamma = angles[0], angles[1], angles[2]
            

        #     device = a.device
        #     dtype = a.dtype

        #     # Create constants on the CORRECT device immediately
        #     zero = torch.tensor(0.0, device=device, dtype=dtype)
        #     # Build cell vectors (standard crystallographic convention)
        #     ax, ay, az = a, zero, zero
        #     bx = b * torch.cos(gamma)
        #     by = b * torch.sin(gamma)
        #     bz = zero
        #     cx = c * torch.cos(beta)
        #     cy = c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / torch.sin(gamma)
        #     cz = torch.sqrt(torch.clamp(c**2 - cx**2 - cy**2, min=1e-8))
            
        #     cell = torch.stack([
        #         torch.stack([ax, ay, az]),
        #         torch.stack([bx, by, bz]),
        #         torch.stack([cx, cy, cz])
        #     ])  # (3, 3)
            
        #     print(f"  Cell matrix (Å):\n{cell}")
        #     print(f"  Cell volume: {torch.linalg.det(cell):.2f} Å³")
            
        #     # Step 2: Convert positions to Ångströms
        #     pos_angstrom = pos * 10.0  # nm to Å
            

        #     # --- THE TRUTH FINDER ---
        #     device = pos.device
        #     cell_inv = torch.linalg.inv(cell)

        #     # 1. Standard (Row-major)
        #     f_row = (pos * 10.0) @ cell_inv.T
        #     # 2. Column-major 
        #     f_col = (torch.linalg.solve(cell, (pos * 10.0).T)).T
        #     # 3. Row-major with 0.5 shift
        #     f_row_shift = f_row + 0.5
        #     # 4. Column-major with 0.5 shift
        #     f_col_shift = f_col + 0.5

        #     conventions = {
        #         "Row-major": f_row,
        #         "Column-major": f_col,
        #         "Row-major + Shift": f_row_shift,
        #         "Column-major + Shift": f_col_shift
        #     }

        #     print(f"\n[DEBUG] Searching for the correct convention:")
        #     for name, f_test in conventions.items():
        #         # We compare the 'span' of the coordinates first
        #         test_span = f_test.max() - f_test.min()
        #         dec_span = frac_from_decoder.max() - frac_from_decoder.min()
        #         diff = (f_test - frac_from_decoder).abs().mean()
                
        #         print(f"  {name:20}: Mean Diff={diff:.4f}, Span Ratio={test_span/dec_span:.2f}")


        #     # Step 3: Compute fractional coordinates
        #     try:
        #         cell_inv = torch.linalg.inv(cell)
        #         frac_manual = pos_angstrom @ cell_inv.T
        #         frac_manual_wrapped = frac_manual % 1.0
                
        #         print(f"  Manually computed frac (unwrapped): [{frac_manual.min():.4f}, {frac_manual.max():.4f}]")
        #         print(f"  Manually computed frac (wrapped):   [{frac_manual_wrapped.min():.4f}, {frac_manual_wrapped.max():.4f}]")
                
        #         # Compare with decoder output
        #         diff = (frac_from_decoder - frac_manual).abs().max()
        #         print(f"  Difference from decoder frac: {diff:.6f}")
                
        #         if diff > 0.1:
        #             print(f"  ⚠️  Large difference! Decoder is computing frac_coords differently!")
                
        #     except RuntimeError as e:
        #         print(f"  ❌ Error computing manual frac_coords: {e}")
        #         print(f"     Cell might be singular or degenerate")
            
        #     start_idx += n

        # print(f"{'='*70}\n")
 
        return out, decoder_batch, samples
 

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
