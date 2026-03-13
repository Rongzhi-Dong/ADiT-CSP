"""
Copyright (c) Meta Platforms, Inc. and affiliates.

MODIFIED FOR CRYSTAL STRUCTURE PREDICTION (CSP):

Summary of changes vs original ldm_module.py:
    1. Removed: multi-dataset support (QM9, QMOF).
       Reason: Crystal-only pipeline — all multi-dataset logic removed.

    2. Changed: forward() passes atom_types to the DiT denoiser.
       Reason: The DiT now conditions on per-token atom types (not a global
       dataset label). We extract atom_types from the batch, convert to the
       dense (B, N_max) format matching the padded latent, and pass them through.

    3. Changed: sample_and_decode() takes atom_types as a required argument.
       Reason: In CSP inference, atom types (composition) are *given*. The caller
       must supply them. The DiT then denoises the structure conditioned on this
       known composition, and the decoder also receives atom_types explicitly.

    4. Removed: dataset_idx and spacegroup conditioning in forward() and sampling.
       Reason: Single domain (crystals). Spacegroup conditioning can be re-added
       later as an adaLN term in the DiT; we keep it out for clarity.

    5. Changed: sample_and_decode() sets encoded_batch['atom_types'] before
       calling autoencoder.decode() so the conditional decoder gets atom types.
       Reason: This is the critical CVAE inference step — the decoder needs
       atom types as condition. Without this, the decoder has no composition signal.
"""

import copy
import os
import time
from typing import Any, Dict, Literal

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

from src.eval.crystal_reconstruction import CrystalReconstructionEvaluator
from src.models.vae_module import VariationalAutoencoderLitModule
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


class CrystalCSPLDMLitModule(LightningModule):
    """LightningModule for Crystal Structure Prediction via latent diffusion.

    Stage 2 of the CSP-ADiT pipeline. The frozen CVAE from Stage 1 encodes
    crystal structures to latents. The DiT learns to denoise those latents
    conditioned on atom types (known composition).

    At inference: given atom types → sample latent → decode → crystal structure.
    """

    def __init__(
        self,
        autoencoder_ckpt: str,
        denoiser: torch.nn.Module,
        interpolant: DictConfig,
        augmentations: DictConfig,
        sampling: DictConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: str,
        compile: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Load and freeze the Stage 1 CVAE
        log.info(f"Loading CVAE checkpoint: {autoencoder_ckpt}")
        self.autoencoder = VariationalAutoencoderLitModule.load_from_checkpoint(
            autoencoder_ckpt, map_location="cpu"
        )
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()

        self.denoiser = denoiser
        self.interpolant = interpolant

        # Crystal CSP evaluator — uses StructureMatcher against gt structures
        self.val_generation_evaluator = CrystalReconstructionEvaluator()
        self.test_generation_evaluator = CrystalReconstructionEvaluator()

        # Metrics
        metric_keys = [
            "loss", "x_loss",
            "x_loss t=[0,25)", "x_loss t=[25,50)", "x_loss t=[50,75)", "x_loss t=[75,100)",
            "t_avg",
        ]
        self.train_metrics = ModuleDict({k: MeanMetric() for k in metric_keys})
        eval_metric_keys = metric_keys + [
            "match_rate@1", "match_rate@20",
            "rms_dist@1", "rms_dist@20", "struct_valid_rate", "sampling_time",
        ]
        self.val_metrics = ModuleDict({k: MeanMetric() for k in eval_metric_keys})
        self.test_metrics = copy.deepcopy(self.val_metrics)

        # Load atom count distribution for sampling
        self.num_nodes_bincount = torch.nn.Parameter(
            torch.load(
                os.path.join(sampling.data_dir, "mp_20/num_nodes_bincount.pt"),
                map_location="cpu",
            ),
            requires_grad=False,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Forward (training)
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, batch: Data, sample_posterior: bool = True):
        """Encode batch to latent, corrupt with noise, run DiT denoiser.

        [CHANGED] atom_types are extracted and passed to the DiT as conditioning.
        """
        with torch.no_grad():
            # --- Stage 1: encode to latent space ---
            encoded_batch = self.autoencoder.encode(batch)
            if sample_posterior:
                encoded_batch["x"] = encoded_batch["posterior"].sample()
            else:
                encoded_batch["x"] = encoded_batch["posterior"].mode()
            x_1 = encoded_batch["x"]  # (N_total, latent_dim)

            # Convert sparse PyG format to dense padded batch: (B, N_max, latent_dim)
            x_1, mask = to_dense_batch(x_1, encoded_batch["batch"])

            # [NEW] Extract atom_types and convert to dense format (B, N_max).
            # This is needed so the DiT receives per-token atom type conditioning.
            # Padding positions get index 0; they are masked out anyway.
            atom_types_sparse = encoded_batch["atom_types"]  # (N_total,)
            atom_types_dense, _ = to_dense_batch(
                atom_types_sparse, encoded_batch["batch"]
            )  # (B, N_max)

            dense_encoded_batch = {
                "x_1": x_1,
                "token_mask": mask,
                "diffuse_mask": mask,
                "atom_types": atom_types_dense,  # [NEW] carry atom types in dense batch
            }

        # --- Stage 2: corrupt with interpolant ---
        self.interpolant.device = dense_encoded_batch["x_1"].device
        noisy_dense_encoded_batch = self.interpolant.corrupt_batch(dense_encoded_batch)

        # --- Self-conditioning (same as original) ---
        if (
            self.interpolant.self_condition
            and torch.rand(1).item() < self.interpolant.self_condition_prob
        ):
            with torch.no_grad():
                x_sc = self.denoiser(
                    x=noisy_dense_encoded_batch["x_t"],
                    t=noisy_dense_encoded_batch["t"],
                    atom_types=dense_encoded_batch["atom_types"],  # [NEW]
                    mask=mask,
                    x_sc=None,
                )
        else:
            x_sc = None

        # --- DiT denoiser forward ---
        # [CHANGED] Pass atom_types instead of dataset_idx + spacegroup
        pred_x = self.denoiser(
            x=noisy_dense_encoded_batch["x_t"],
            t=noisy_dense_encoded_batch["t"],
            atom_types=dense_encoded_batch["atom_types"],  # [NEW] per-token conditioning
            mask=mask,
            x_sc=x_sc,
        )

        return pred_x, noisy_dense_encoded_batch

    # ─────────────────────────────────────────────────────────────────────────
    # Loss (unchanged from original logic)
    # ─────────────────────────────────────────────────────────────────────────

    def criterion(
        self,
        noisy_dense_encoded_batch: Dict[str, torch.Tensor],
        pred_x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        gt_x_1 = noisy_dense_encoded_batch["x_1"]
        norm_scale = 1 - torch.min(noisy_dense_encoded_batch["t"].unsqueeze(-1), torch.tensor(0.9))
        x_error = (gt_x_1 - pred_x) / norm_scale
        loss_mask = (
            noisy_dense_encoded_batch["token_mask"] * noisy_dense_encoded_batch["diffuse_mask"]
        )
        loss_denom = torch.sum(loss_mask, dim=-1) * pred_x.size(-1)
        x_loss = torch.sum(x_error**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        loss_dict = {"loss": x_loss.mean(), "x_loss": x_loss}

        # Stratify loss across diffusion timesteps for monitoring
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
            loss_dict[t_range] = t_binned_loss[t_bin] / t_binned_n[t_bin]
        loss_dict["t_avg"] = np.mean(flat_t)
        return loss_dict

    # ─────────────────────────────────────────────────────────────────────────
    # Training step
    # ─────────────────────────────────────────────────────────────────────────

    def on_train_start(self) -> None:
        for metric in self.val_metrics.values():
            metric.reset()

    def on_train_epoch_start(self) -> None:
        for metric in self.train_metrics.values():
            metric.reset()

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        with torch.no_grad():
            # Fractional coordinate translation augmentation.
            # Randomly shift all atoms by the same vector — a symmetry of periodic crystals.
            # NOTE: batch.pos is intentionally not used; the flow matching model operates
            # entirely in frac_coords + lattice space.
            if self.hparams.augmentations.frac_coords:
                random_shift = torch.rand(3, device=self.device)  # uniform in [0,1)^3
                batch.frac_coords = (batch.frac_coords + random_shift) % 1.0
            # NOTE: rotation augmentation removed — batch.pos is not used by the CSP
            # encoder/decoder, so rotating it has no effect on training.

        pred_x, noisy_dense_encoded_batch = self.forward(batch)
        loss_dict = self.criterion(noisy_dense_encoded_batch, pred_x)

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
        self.val_generation_evaluator.clear()
        self.val_atom_types_buffer = []  # collect real compositions for CSP eval
        self.val_gt_buffer = []  # collect gt structures for match_rate

    def validation_step(self, batch: Data, batch_idx: int) -> None:
        self._evaluation_step(batch, batch_idx, stage="val")

    def on_validation_epoch_end(self) -> None:
        self._on_evaluation_epoch_end(stage="val")

    def on_test_epoch_start(self) -> None:
        for metric in self.test_metrics.values():
            metric.reset()
        self.test_generation_evaluator.clear()
        self.test_atom_types_buffer = []  # collect real compositions for CSP eval
        self.test_gt_buffer = []  # collect gt structures for match_rate

    def test_step(self, batch: Data, batch_idx: int) -> None:
        self._evaluation_step(batch, batch_idx, stage="test")

    def on_test_epoch_end(self) -> None:
        self._on_evaluation_epoch_end(stage="test")

    def _evaluation_step(self, batch: Data, batch_idx: int, stage: str) -> None:
        metrics = self.val_metrics if stage == "val" else self.test_metrics

        # [CSP FIX] Collect real per-sample atom_types for use in evaluation sampling.
        # This is the key fix for valid_rate=0: instead of sampling with all-zero
        # atom_types (no conditioning), we use real compositions from the val/test set.
        buffer = self.val_atom_types_buffer if stage == "val" else self.test_atom_types_buffer
        gt_buffer = self.val_gt_buffer if stage == "val" else self.test_gt_buffer
        for i in range(batch.num_graphs):
            mask = batch.batch == i
            buffer.append(batch.atom_types[mask].cpu())
            num_atoms_i = mask.sum().item()
            gt_buffer.append({
                "atom_types": batch.atom_types[mask].cpu().numpy(),
                "frac_coords": batch.frac_coords[mask].cpu().numpy(),
                "lengths": (batch.lengths_scaled[i] * num_atoms_i ** (1/3)).cpu().numpy(),
                "angles": torch.rad2deg(batch.angles_radians[i]).cpu().numpy(),
            })

        pred_x, noisy_dense_encoded_batch = self.forward(batch)
        loss_dict = self.criterion(noisy_dense_encoded_batch, pred_x)
        for k, v in loss_dict.items():
            metrics[k](v)
            self.log(f"{stage}/{k}", metrics[k], on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=True)

    def _on_evaluation_epoch_end(self, stage: str) -> None:
        """
        CSP evaluation with K candidates per composition.

        For each of num_samples compositions drawn from val/test set:
          - Generate num_candidates independent structure predictions
          - match_rate@K = fraction of compositions where ANY of first K candidates matched gt
          - match_rate@1  → standard single-shot metric
          - match_rate@20 → best-of-20 (upper bound, publication standard)
        """
        import random

        metrics   = self.val_metrics   if stage == "val" else self.test_metrics
        buffer    = self.val_atom_types_buffer if stage == "val" else self.test_atom_types_buffer
        gt_buffer = self.val_gt_buffer if stage == "val" else self.test_gt_buffer

        num_samples    = self.hparams.sampling.num_samples     # number of compositions
        num_candidates = self.hparams.sampling.num_candidates  # K candidates per composition
        batch_size     = self.hparams.sampling.batch_size

        # Sample paired (composition, gt) from the buffer — must stay paired!
        # Both buffer and gt_buffer are aligned: buffer[i] and gt_buffer[i]
        # are atom_types and full gt structure for the SAME crystal.
        n_available = len(gt_buffer)
        if num_samples <= n_available:
            indices = random.sample(range(n_available), num_samples)
        else:
            # Cycle if num_samples > val set size
            indices = list(range(n_available)) * (num_samples // n_available + 1)
            indices = indices[:num_samples]
        sampled_compositions = [buffer[i] for i in indices]
        sampled_gts          = [gt_buffer[i] for i in indices]

        # ── Generate K candidates per composition ──────────────────────────
        # all_preds[k] = list of num_samples pred dicts for candidate k
        all_preds = [[] for _ in range(num_candidates)]

        t_start = time.time()
        for k in range(num_candidates):
            evaluator_k = CrystalReconstructionEvaluator()
            for i in tqdm(
                range(0, num_samples, batch_size),
                desc=f"Sampling crystals — candidate {k+1}/{num_candidates}",
                leave=False,
            ):
                batch_atom_types  = sampled_compositions[i:i + batch_size]
                actual_batch_size = len(batch_atom_types)

                out, batch_info = self.sample_and_decode(
                    num_nodes_bincount=self.num_nodes_bincount,
                    batch_size=actual_batch_size,
                    cfg_scale=self.hparams.sampling.cfg_scale,
                    atom_types_per_sample=batch_atom_types,
                )

                start_idx = 0
                for idx_in_batch, num_atom in enumerate(batch_info["num_atoms"].tolist()):
                    global_idx = i + idx_in_batch
                    pred = {
                        "atom_types":  batch_info["atom_types"][start_idx: start_idx + num_atom].detach().cpu().numpy(),
                        "frac_coords": out["frac_coords"].narrow(0, start_idx, num_atom).detach().cpu().numpy(),
                        "lengths":     (out["lengths"][idx_in_batch] * float(num_atom) ** (1/3)).detach().cpu().numpy(),
                        "angles":      torch.rad2deg(out["angles"][idx_in_batch]).detach().cpu().numpy(),
                        "sample_idx":  global_idx,
                    }
                    all_preds[k].append(pred)
                    evaluator_k.append_pred_array(pred)
                    evaluator_k.append_gt_array({**sampled_gts[global_idx], "sample_idx": global_idx})
                    start_idx += num_atom

            # Compute metrics for candidate k
            metrics_k = evaluator_k.get_metrics(
                save=(self.hparams.sampling.visualize and k == 0),
                save_dir=self.hparams.sampling.save_dir + f"/{stage}_{self.global_rank}/cand{k}",
            )
            # Store per-candidate match arrays and rms arrays: shape (num_samples,)
            if k == 0:
                match_arrays        = [metrics_k["match_rate"].bool()]
                rms_arrays          = [metrics_k["rms_dist"]]
                rms_k0              = metrics_k["rms_dist"]
                evaluator_for_table = evaluator_k
            else:
                match_arrays.append(metrics_k["match_rate"].bool())
                rms_arrays.append(metrics_k["rms_dist"])

        t_end = time.time()

        # ── Compute match_rate@K ───────────────────────────────────────────
        # match_arrays: list of K tensors, each (num_samples,) bool
        # match_rate@K = fraction of samples where ANY of first K candidates matched
        match_stack = torch.stack(match_arrays, dim=0)  # (K, num_samples)

        def match_at_k(stack, k):
            """Fraction of samples matched by any of first k candidates."""
            return stack[:k].any(dim=0).float().mean()

        mr1  = match_at_k(match_stack, 1)
        mr20 = match_at_k(match_stack, min(20, num_candidates))

        rms_dist_val = rms_k0.mean() if len(rms_k0) > 0 else torch.tensor(float("inf"))

        # rms_dist@20: for each sample, take the MIN rms across all 20 candidates that matched,
        # then average those per-sample minimums. This exactly matches DiffCSP RecEvalBatch:
        #   rms_dists.append(np.min(tmp_rms_dists))   <- min per structure
        #   mean_rms_dist = rms_dists[...].mean()      <- then mean
        k20 = min(20, num_candidates)
        per_sample_min_rms = torch.full((num_samples,), float("inf"))
        for k_idx in range(k20):
            mk = match_stack[k_idx]                    # (num_samples,) bool
            rk = rms_arrays[k_idx]                     # RMSD for matched samples in cand k
            if not (isinstance(rk, torch.Tensor) and len(rk) > 0):
                continue
            matched_indices = mk.nonzero(as_tuple=True)[0]
            if len(matched_indices) != len(rk):
                continue
            for j, sample_j in enumerate(matched_indices.tolist()):
                per_sample_min_rms[sample_j] = torch.minimum(
                    per_sample_min_rms[sample_j], rk[j])
        valid_min = per_sample_min_rms[per_sample_min_rms < float("inf")]
        rms_dist_20_val = valid_min.mean() if len(valid_min) > 0 else torch.tensor(float("inf"))

        # struct_valid_rate from candidate 0
        pred_list = evaluator_for_table.pred_crys_list
        gt_list   = evaluator_for_table.gt_crys_list
        struct_valid_rate = torch.tensor(
            sum(c.struct_valid for c in pred_list) / len(pred_list)
        ) if pred_list else torch.tensor(0.0)

        # Sanity check
        if pred_list and gt_list:
            comp_match_rate = sum(
                sorted(p.atom_types) == sorted(g.atom_types)
                for p, g in zip(pred_list, gt_list)
            ) / len(pred_list)
            if comp_match_rate < 1.0:
                log.warning(f"comp_match_rate={comp_match_rate:.4f} < 1.0 — model is not preserving input atom types!")

        # ── WandB table (candidate 0 only) ────────────────────────────────
        if self.hparams.sampling.visualize:
            all_loggers = self.loggers if hasattr(self, "loggers") and self.loggers else ([self.logger] if self.logger else [])
            wandb_logger = next((l for l in all_loggers if isinstance(l, WandbLogger)), None)
            if wandb_logger is not None:
                pred_table = evaluator_for_table.get_wandb_table(
                    current_epoch=self.current_epoch,
                    save_dir=self.hparams.sampling.save_dir + f"/{stage}_{self.global_rank}/cand0",
                )
                wandb_logger.experiment.log(
                    {f"mp20_{stage}_samples_table_device{self.global_rank}": pred_table}
                )

        # ── Log metrics ───────────────────────────────────────────────────
        self.log(f"{stage}/match_rate@1",      mr1,               on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log(f"{stage}/match_rate@20",     mr20,              on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log(f"{stage}/rms_dist@1",        rms_dist_val,      on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{stage}/rms_dist@20",       rms_dist_20_val,   on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{stage}/struct_valid_rate", struct_valid_rate,  on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log(f"{stage}/sampling_time",     torch.tensor(t_end - t_start), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    # ─────────────────────────────────────────────────────────────────────────
    # CSP Sampling
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample_and_decode(
        self,
        num_nodes_bincount: torch.Tensor,
        batch_size: int,
        cfg_scale: float = 1.0,
        atom_types_per_sample: torch.Tensor = None,
    ):
        """Sample crystal structures given (optionally) known atom types.

        [CHANGED] atom_types_per_sample is now a first-class argument.
        If provided (CSP mode): use given compositions for conditioning.
        If None (generative mode): sample random atom counts and use null conditioning.

        Args:
            num_nodes_bincount: Distribution over number of atoms per crystal
            batch_size: Number of crystals to generate
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
            atom_types_per_sample: Optional list of atom_type tensors, one per sample.
                Each is a 1D tensor of atomic numbers. Length must equal batch_size.
                If None, random atom counts are used with null atom type conditioning.

        Returns:
            out: dict with frac_coords, lengths, angles
            batch_info: dict with num_atoms, batch, token_idx, atom_types
        """
        # --- Determine per-sample atom counts ---
        if atom_types_per_sample is not None:
            # CSP mode: compositions are given
            assert len(atom_types_per_sample) == batch_size
            sample_lengths = torch.tensor(
                [a.shape[0] for a in atom_types_per_sample],
                dtype=torch.long, device=self.device,
            )
            # Build sparse atom_types and batch index
            atom_types_sparse = torch.cat(
                [a.to(self.device) for a in atom_types_per_sample]
            )  # (N_total,)
            batch_idx_sparse = torch.repeat_interleave(
                torch.arange(batch_size, device=self.device), sample_lengths
            )
        else:
            # Generative mode: sample random lengths, no composition conditioning
            sample_lengths = torch.multinomial(
                num_nodes_bincount.float(), batch_size, replacement=True
            ).to(self.device)
            atom_types_sparse = None
            batch_idx_sparse = None

        max_len = int(sample_lengths.max().item())

        # --- Build token mask ---
        token_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)
        for idx, length in enumerate(sample_lengths):
            token_mask[idx, :length] = True

        # --- Build dense atom type tensor for DiT conditioning ---
        # Shape: (B, N_max); padding positions get 0 (masked out anyway)
        if atom_types_sparse is not None:
            atom_types_dense, _ = to_dense_batch(atom_types_sparse, batch_idx_sparse)
            # Pad to max_len if needed
            if atom_types_dense.shape[1] < max_len:
                pad = torch.zeros(
                    batch_size, max_len - atom_types_dense.shape[1],
                    dtype=torch.long, device=self.device,
                )
                atom_types_dense = torch.cat([atom_types_dense, pad], dim=1)
        else:
            # Null conditioning: all zeros (unknown composition)
            atom_types_dense = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)

        # --- Run DiT ODE solver ---
        # [CHANGED] Pass atom_types_dense to the interpolant sampler.
        # The interpolant's sample_with_classifier_free_guidance must call
        # denoiser(x, t, atom_types=..., mask=...) — update interpolant accordingly.
        samples = self.interpolant.sample_with_classifier_free_guidance(
            batch_size=batch_size,
            num_tokens=max_len,
            emb_dim=self.denoiser.d_x,
            model=self.denoiser,
            atom_types=atom_types_dense,  # [NEW] replaces dataset_idx + spacegroup
            cfg_scale=cfg_scale,
            token_mask=token_mask,
        )

        # Extract final denoised latent (remove padding)
        x = samples["clean_traj"][-1][token_mask]  # (N_total, latent_dim)

        # Build batch info dict
        if atom_types_sparse is None:
            # No composition given — use zeros as placeholder
            atom_types_sparse = torch.zeros(
                token_mask.sum().item(), dtype=torch.long, device=self.device
            )
            batch_idx_sparse = torch.repeat_interleave(
                torch.arange(batch_size, device=self.device), sample_lengths
            )

        token_idx_sparse = (torch.cumsum(token_mask, dim=-1, dtype=torch.int64) - 1)[token_mask]

        batch_info = {
            "x": x,
            "num_atoms": sample_lengths,
            "batch": batch_idx_sparse,
            "token_idx": token_idx_sparse,
            "atom_types": atom_types_sparse,  # [NEW] carries composition for decoder
        }

        # --- Decode latent to crystal structure ---
        # [CHANGED] We must set atom_types in encoded_batch BEFORE calling decode().
        # This is the CVAE inference step: decode(z, composition) → structure.
        # The decoder needs atom types as its conditioning signal.
        encoded_batch_for_decode = {
            "x": x,
            "num_atoms": sample_lengths,
            "batch": batch_idx_sparse,
            "token_idx": token_idx_sparse,
            "atom_types": atom_types_sparse,  # [KEY] conditional decoder requires this
        }
        out = self.autoencoder.decode(encoded_batch_for_decode)

        return out, batch_info

    # ─────────────────────────────────────────────────────────────────────────
    # Setup / Optimizers
    # ─────────────────────────────────────────────────────────────────────────

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.autoencoder = torch.compile(self.autoencoder)
            self.denoiser = torch.compile(self.denoiser)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/match_rate@1",
                    "interval": "epoch",
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}