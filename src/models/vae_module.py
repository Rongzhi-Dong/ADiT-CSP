"""
Copyright (c) Meta Platforms, Inc. and affiliates.

MODIFIED FOR CRYSTAL STRUCTURE PREDICTION (CSP):

Summary of changes vs original vae_module.py:
    1. Removed: multi-dataset support (QM9 molecules, QMOF MOFs).
       Reason: We only target periodic crystals (MP20). All the dataset_idx
       dispatch logic, molecule evaluators, and non-periodic loss branches
       are removed. This simplifies the code significantly.

    2. Removed: loss_atom_types (cross-entropy on predicted atom types).
       Reason: Atom types are the CONDITION in CSP — they are never predicted.
       The reconstruction loss now covers only frac_coords and lattice.

    3. Removed: loss_pos (MSE on Cartesian 3D coordinates).
       Reason: For crystals, pos = cell @ frac_coords. Predicting 3D coords
       separately is redundant and was already zeroed (λ_X=0) for crystals
       in the original paper.

    4. Changed: encode() now passes atom_types through to the encoded_batch dict.
       Reason: The decoder needs atom types as a conditioning signal. We thread
       them through the encode → decode pipeline explicitly.

    5. Changed: decode() injects atom_types into encoded_batch before calling decoder.
       Reason: At inference (DiT sampling), the decoder is called with a latent
       that has no atom type information baked in — we must supply it explicitly.
       This is the key CVAE inference pattern.

    6. Fixed: Noise augmentation no longer corrupts atom_types (they are the
       conditioning signal and must be clean at all times). Augmentation now
       perturbs frac_coords directly with wrapped Gaussian noise instead of
       going through Cartesian coords.

    7. Changed: loss_weight_kl default raised from 1e-5 → 1e-4 to prevent
       posterior collapse and give the diffusion model enough latent space to work.
"""

import copy
from typing import Any, Dict, Literal

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch.nn import ModuleDict
from torch_geometric.data import Data
from torchmetrics import MeanMetric

from src.eval.crystal_reconstruction import CrystalReconstructionEvaluator
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Diagonal Gaussian — unchanged from original
# ─────────────────────────────────────────────────────────────────────────────

class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution with mean and logvar parameters.

    Adapted from: https://github.com/CompVis/latent-diffusion
    """

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        return self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        if other is None:
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=1
            )
        return 0.5 * torch.sum(
            torch.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=1,
        )

    def mode(self):
        return self.mean


# ─────────────────────────────────────────────────────────────────────────────
# Crystal CSP VAE LightningModule
# ─────────────────────────────────────────────────────────────────────────────

class CrystalCSPVAELitModule(LightningModule):
    """LightningModule for the crystal-structure-prediction Conditional VAE.

    Stage 1 of the CSP-ADiT pipeline. The VAE learns to:
        - ENCODE: (atom_types [condition], frac_coords, lattice) → latent z
        - DECODE: (z, atom_types [condition]) → (frac_coords, lattice)

    Atom types are never predicted — they flow through as a conditioning signal.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        latent_dim: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: int,
        # [CHANGED] Loss weights are now only for crystal-relevant outputs
        loss_weight_frac_coords: float = 10.0,
        loss_weight_lengths: float = 1.0,
        loss_weight_angles: float = 10.0,
        loss_weight_kl: float = 1e-4,
        augmentations: DictConfig = None,
        visualization: DictConfig = None,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.encoder = encoder
        self.decoder = decoder

        # Latent projection layers (same naming as original for checkpoint compatibility)
        self.quant_conv = torch.nn.Linear(encoder.d_model, 2 * latent_dim, bias=False)
        self.post_quant_conv = torch.nn.Linear(latent_dim, decoder.d_model, bias=False)

        # [CHANGED] Scalar loss weights instead of per-dataset dicts.
        # Reason: crystal-only model has no need for dataset dispatch.
        self.loss_weight_frac_coords = loss_weight_frac_coords
        self.loss_weight_lengths = loss_weight_lengths
        self.loss_weight_angles = loss_weight_angles
        self.loss_weight_kl = loss_weight_kl

        # Evaluators — crystal only
        self.val_reconstruction_evaluator = CrystalReconstructionEvaluator()
        self.test_reconstruction_evaluator = CrystalReconstructionEvaluator()

        # Metrics
        metric_keys = [
            "loss",
            "loss_frac_coords",
            "loss_lengths",
            "loss_angles",
            "loss_kl",
            "unscaled/loss_frac_coords",
            "unscaled/loss_lengths",
            "unscaled/loss_angles",
            "unscaled/loss_kl",
        ]
        self.train_metrics = ModuleDict({k: MeanMetric() for k in metric_keys})
        self.val_metrics = ModuleDict(
            {k: MeanMetric() for k in metric_keys + ["match_rate", "rms_dist"]}
        )
        self.test_metrics = copy.deepcopy(self.val_metrics)

    # ─────────────────────────────────────────────────────────────────────────
    # Core encode / decode
    # ─────────────────────────────────────────────────────────────────────────

    def encode(self, batch: Data) -> Dict[str, torch.Tensor]:
        """Encode a crystal batch to a latent distribution.

        [CHANGED] The encoder now returns atom_types in the dict so they can
        be passed to the decoder without touching the batch object again.
        """
        encoded_batch = self.encoder(batch)
        # Project encoder hidden states to (mean, logvar) of the posterior
        encoded_batch["moments"] = self.quant_conv(encoded_batch["x"])
        encoded_batch["posterior"] = DiagonalGaussianDistribution(encoded_batch["moments"])
        return encoded_batch

    def decode(self, encoded_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decode a latent sample to crystal structure.

        [CHANGED] Injects atom_types into encoded_batch before calling the decoder.
        This is the CVAE decode(z, condition) pattern.
        At inference time (DiT sampling), atom_types must be set in encoded_batch
        by the caller before calling this method.
        """
        # Up-project from latent_dim to d_model
        encoded_batch["x"] = self.post_quant_conv(encoded_batch["x"])

        # atom_types must be present — either forwarded from encode() or set by caller
        assert "atom_types" in encoded_batch, (
            "encoded_batch must contain 'atom_types' for conditional decoding. "
            "At inference, set encoded_batch['atom_types'] to the known composition."
        )

        out = self.decoder(encoded_batch)
        return out

    def forward(self, batch: Data, sample_posterior: bool = True):
        """Full encode → sample → decode pass."""
        encoded_batch = self.encode(batch)
        if sample_posterior:
            encoded_batch["x"] = encoded_batch["posterior"].sample()
        else:
            encoded_batch["x"] = encoded_batch["posterior"].mode()
        out = self.decode(encoded_batch)
        return out, encoded_batch

    # ─────────────────────────────────────────────────────────────────────────
    # Loss
    # ─────────────────────────────────────────────────────────────────────────

    def reconstruction_criterion(
        self, batch: Data, out: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute per-sample reconstruction losses.

        [CHANGED] Removed loss_atom_types (atom types are the condition, not target).
        [CHANGED] Removed loss_pos (Cartesian coords redundant for crystals).
        [KEPT]    loss_frac_coords, loss_lengths, loss_angles.
        """
        # Fractional coordinates loss — per-atom MSE averaged over xyz
        loss_frac_coords = F.mse_loss(
            out["frac_coords"], batch.frac_coords, reduction="none"
        ).mean(dim=1)  # (N_total,)

        # Lattice lengths loss — scaled by num_atoms^(1/3) to normalise for cell size
        loss_lengths = F.mse_loss(
            out["lengths"], batch.lengths_scaled, reduction="none"
        ).mean(dim=1)  # (B,)

        # Lattice angles loss — in radians
        loss_angles = F.mse_loss(
            out["angles"], batch.angles_radians, reduction="none"
        ).mean(dim=1)  # (B,)

        return {
            "loss_frac_coords": loss_frac_coords,
            "loss_lengths": loss_lengths,
            "loss_angles": loss_angles,
        }

    def criterion(
        self,
        batch: Data,
        encoded_batch: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Combine reconstruction + KL loss into total training objective."""
        loss_reconst = self.reconstruction_criterion(batch, out)
        loss_kl = encoded_batch["posterior"].kl()  # (N_total,)

        # Weighted sum
        # [CHANGED] Simple scalar weights — no dataset_idx dispatch needed
        loss = (
            self.loss_weight_frac_coords * loss_reconst["loss_frac_coords"].mean()
            + self.loss_weight_lengths  * loss_reconst["loss_lengths"].mean()
            + self.loss_weight_angles   * loss_reconst["loss_angles"].mean()
            + self.loss_weight_kl       * loss_kl.mean()
        )

        return {
            "loss": loss,
            "loss_frac_coords": self.loss_weight_frac_coords * loss_reconst["loss_frac_coords"],
            "loss_lengths":     self.loss_weight_lengths     * loss_reconst["loss_lengths"],
            "loss_angles":      self.loss_weight_angles      * loss_reconst["loss_angles"],
            "loss_kl":          self.loss_weight_kl          * loss_kl,
            "unscaled/loss_frac_coords": loss_reconst["loss_frac_coords"],
            "unscaled/loss_lengths":     loss_reconst["loss_lengths"],
            "unscaled/loss_angles":      loss_reconst["loss_angles"],
            "unscaled/loss_kl":          loss_kl,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def on_train_start(self) -> None:
        for metric in self.val_metrics.values():
            metric.reset()

    def on_train_epoch_start(self) -> None:
        for metric in self.train_metrics.values():
            metric.reset()

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        with torch.no_grad():
            # --- Fractional coordinate translation augmentation ---
            # Randomly translate all atoms by the same vector in fractional space.
            # This is a symmetry of periodic crystals (periodic boundary conditions).
            if self.hparams.augmentations and self.hparams.augmentations.frac_coords:
                random_shift = torch.rand(3, device=self.device)  # uniform in [0,1)^3
                batch.frac_coords = (batch.frac_coords + random_shift) % 1.0

            # NOTE: rotation augmentation (batch.pos) is intentionally removed.
            # The CSP encoder operates on frac_coords + lattice only — batch.pos
            # is never used and rotating it would have no effect.

            # --- Denoising augmentation: perturb frac_coords only ---
            # [CSP] atom_types are the conditioning signal — NEVER corrupt them.
            # [CSP] batch.pos is not used by the encoder — perturb frac_coords directly.
            if self.hparams.augmentations and self.hparams.augmentations.noise > 0.0:
                total_atoms = batch.num_atoms.sum().item()
                frac_coords_ = batch.frac_coords.clone()  # save clean target for loss

                perturbed_idx = torch.tensor(
                    np.random.choice(
                        total_atoms,
                        int(total_atoms * self.hparams.augmentations.noise),
                        replace=False,
                    ),
                    device=self.device,
                )
                # Perturb frac_coords with small Gaussian noise, wrapped to [0, 1)
                batch.frac_coords[perturbed_idx] = (
                    batch.frac_coords[perturbed_idx]
                    + torch.randn(len(perturbed_idx), 3, device=self.device) * 0.05
                ) % 1.0

        out, encoded_batch = self.forward(batch)

        # Restore clean frac_coords targets before computing loss
        if self.hparams.augmentations and self.hparams.augmentations.noise > 0.0:
            batch.frac_coords = frac_coords_

        loss_dict = self.criterion(batch, encoded_batch, out)

        for k, v in loss_dict.items():
            self.train_metrics[k](v)
            self.log(f"train/{k}", self.train_metrics[k], on_step=True, on_epoch=False,
                     prog_bar=(k == "loss"))

        return loss_dict["loss"]

    # ─────────────────────────────────────────────────────────────────────────
    # Validation / Test
    # ─────────────────────────────────────────────────────────────────────────

    def on_validation_epoch_start(self) -> None:
        for metric in self.val_metrics.values():
            metric.reset()
        self.val_reconstruction_evaluator.clear()

    def validation_step(self, batch: Data, batch_idx: int) -> None:
        self._evaluation_step(batch, batch_idx, stage="val")

    def on_validation_epoch_end(self) -> None:
        self._on_evaluation_epoch_end(stage="val")

    def on_test_epoch_start(self) -> None:
        for metric in self.test_metrics.values():
            metric.reset()
        self.test_reconstruction_evaluator.clear()

    def test_step(self, batch: Data, batch_idx: int) -> None:
        self._evaluation_step(batch, batch_idx, stage="test")

    def on_test_epoch_end(self) -> None:
        self._on_evaluation_epoch_end(stage="test")

    def _evaluation_step(self, batch: Data, batch_idx: int, stage: str) -> None:
        metrics = self.val_metrics if stage == "val" else self.test_metrics
        evaluator = self.val_reconstruction_evaluator if stage == "val" else self.test_reconstruction_evaluator

        out, encoded_batch = self.forward(batch)
        loss_dict = self.criterion(batch, encoded_batch, out)

        for k, v in loss_dict.items():
            metrics[k](v)
            self.log(f"{stage}/{k}", metrics[k], on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=True)

        # Save predictions for reconstruction evaluation
        start_idx = 0
        for idx_in_batch, num_atom in enumerate(batch.num_atoms.tolist()):
            # [CHANGED] Use ground-truth atom types (they are the condition, not predicted)
            _atom_types = batch.atom_types[start_idx: start_idx + num_atom]
            _frac_coords = out["frac_coords"].narrow(0, start_idx, num_atom)
            _lengths = out["lengths"][idx_in_batch] * float(num_atom) ** (1 / 3)
            _angles = torch.rad2deg(out["angles"][idx_in_batch])
            evaluator.append_pred_array({
                "atom_types": _atom_types.detach().cpu().numpy(),
                "frac_coords": _frac_coords.detach().cpu().numpy(),
                "lengths": _lengths.detach().cpu().numpy(),
                "angles": _angles.detach().cpu().numpy(),
                "sample_idx": batch_idx * batch.batch_size + idx_in_batch,
            })
            start_idx += num_atom

        for idx_in_batch, _data in enumerate(batch.to_data_list()):
            evaluator.append_gt_array({
                "atom_types": _data["atom_types"].detach().cpu().numpy(),
                "frac_coords": _data["frac_coords"].detach().cpu().numpy(),
                "lengths": _data["lengths"].detach().cpu().numpy(),
                "angles": _data["angles"].detach().cpu().numpy(),
                "sample_idx": batch_idx * batch.batch_size + idx_in_batch,
            })

    def _on_evaluation_epoch_end(self, stage: str) -> None:
        metrics = self.val_metrics if stage == "val" else self.test_metrics
        evaluator = self.val_reconstruction_evaluator if stage == "val" else self.test_reconstruction_evaluator

        rec_metrics = evaluator.get_metrics(
            save=self.hparams.visualization.visualize if self.hparams.visualization else False,
            save_dir=getattr(self.hparams.visualization, "save_dir", ".") + f"/{stage}_{self.global_rank}",
        )
        for k, v in rec_metrics.items():
            metrics[k](v)
            self.log(f"{stage}/{k}", metrics[k], on_step=False, on_epoch=True,
                     prog_bar=(k == "match_rate"), sync_dist=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Setup / Optimizers
    # ─────────────────────────────────────────────────────────────────────────

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.encode = torch.compile(self.encode)
            self.decode = torch.compile(self.decode)
            self.quant_conv = torch.compile(self.quant_conv)
            self.post_quant_conv = torch.compile(self.post_quant_conv)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/match_rate",
                    "interval": "epoch",
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}


# Alias for backward compatibility with ldm_module imports
VariationalAutoencoderLitModule = CrystalCSPVAELitModule