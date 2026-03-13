"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import copy

import torch


class FlowMatchingInterpolant:
    """Interpolant for simple Gaussian flow matching.

    - Constructs noisy samples from clean samples during training.
    - Implements sampling loop from random noise (or prior) during inference.
    - Also supports classifier-free guidance for DiT denoisers.

    Adapted from: https://github.com/microsoft/protein-frame-flow

    Args:
        min_t (float): Minimum time step to sample during training.
        corrupt (bool): Whether to corrupt samples during training.
        num_timesteps (int): Number of timesteps to integrate over.
        self_condition (bool): Whether to use self-conditioning during denoising.
        self_condition_prob (float): Probability of using self-conditioning during training.
        device (str): Device to run on.
    """

    def __init__(
        self,
        min_t: int = 1e-2,
        corrupt: bool = True,
        num_timesteps: int = 100,
        self_condition: bool = False,
        self_condition_prob: float = 0.5,
        device: str = "cpu",
    ):
        self.min_t = min_t
        self.corrupt = corrupt
        self.num_timesteps = num_timesteps
        self.self_condition = self_condition
        self.self_condition_prob = self_condition_prob
        self.device = device

    def _sample_t(self, batch_size):
        t = torch.rand(batch_size, device=self.device)
        return t * (1 - 2 * self.min_t) + self.min_t

    def _centered_gaussian(self, batch_size, num_tokens, emb_dim=3):
        noise = torch.randn(batch_size, num_tokens, emb_dim, device=self.device)
        return noise - torch.mean(noise, dim=-2, keepdims=True)

    def _corrupt_x(self, x_1, t, token_mask, diffuse_mask):
        x_0 = self._centered_gaussian(*x_1.shape)
        x_t = (1 - t[..., None]) * x_0 + t[..., None] * x_1
        x_t = x_t * diffuse_mask[..., None] + x_1 * (~diffuse_mask[..., None])
        return x_t * token_mask[..., None]

    def corrupt_batch(self, batch):
        """Corrupts a batch of data by sampling a time t and interpolating to noisy samples.

        Args:
            batch (dict): Batch of clean data with keys:
                - x_1 (torch.Tensor): Clean data tensor.
                - token_mask (torch.Tensor): True if valid token, False if padding.
                - diffuse_mask (torch.Tensor): True if diffusion is to be performed, False if fixed during denoising.
        """
        noisy_batch = copy.deepcopy(batch)

        # [B, N, d]
        x_1 = batch["x_1"]

        # [B, N]
        token_mask = batch["token_mask"]
        diffuse_mask = batch["diffuse_mask"]
        batch_size, _ = diffuse_mask.shape

        # [B, 1]
        t = self._sample_t(batch_size)[:, None]
        noisy_batch["t"] = t

        # Apply corruptions
        if self.corrupt:
            x_t = self._corrupt_x(x_1, t, token_mask, diffuse_mask)
        else:
            x_t = x_1
        if torch.any(torch.isnan(x_t)):
            raise ValueError("NaN in x_t during corruption")
        noisy_batch["x_t"] = x_t

        return noisy_batch

    def _x_vector_field(self, t, x_1, x_t):
        return (x_1 - x_t) / (1 - t)

    def _x_euler_step(self, d_t, t, x_1, x_t):
        assert d_t > 0
        x_vf = self._x_vector_field(t, x_1, x_t)
        return x_t + x_vf * d_t

    def sample(
        self,
        batch_size,
        num_tokens,
        emb_dim,
        model,
        atom_types,
        num_timesteps=None,
        x_0=None,
        x_1=None,
        token_mask=None,
        token_idx=None,
    ):
        """Generates new samples of a specified (B, N, d) using denoiser model.

        CHANGED: replaced dataset_idx + spacegroup args with atom_types (B, N).
        Reason: the DiT denoiser now conditions on per-token atom types instead
        of a global dataset label. The call signature mirrors the new DiT.forward().

        Args:
            batch_size (int): Number of samples to generate.
            num_tokens (int): Number of tokens in each sample.
            emb_dim (int): Dimension of each token.
            model (nn.Module): Denoiser model (CrystalCSPDiT).
            atom_types (torch.Tensor): Atomic numbers per token (B, N). CONDITION.
            num_timesteps (int): Number of ODE integration steps.
            x_0 (torch.Tensor): Initial noise sample (B, N, d). If None, sampled fresh.
            x_1 (torch.Tensor): Clean target (only needed if corrupt=False).
            token_mask (torch.Tensor): True for valid tokens, False for padding (B, N).
            token_idx (torch.Tensor): Token position indices (B, N).

        Returns:
            Dict with keys:
                tokens_traj (list): Noisy latent at each ODE step.
                clean_traj (list): Denoised prediction at each ODE step.
        """
        if x_0 is None:
            x_0 = self._centered_gaussian(batch_size, num_tokens, emb_dim)
        if token_mask is None:
            token_mask = torch.ones(batch_size, num_tokens, device=self.device).bool()
        if token_idx is None:
            token_idx = torch.arange(num_tokens, device=self.device, dtype=torch.float32)[
                None
            ].repeat(batch_size, 1)

        if num_timesteps is None:
            num_timesteps = self.num_timesteps
        ts = torch.linspace(self.min_t, 1.0, num_timesteps)
        t_1 = ts[0]

        tokens_traj = [x_0]
        clean_traj = []
        x_sc = None
        for t_2 in ts[1:]:
            x_t_1 = tokens_traj[-1]
            x = x_t_1 if self.corrupt else x_1
            if not self.corrupt and x_1 is None:
                raise ValueError("Must provide x_1 if not corrupting.")
            t = torch.ones((batch_size, 1), device=self.device) * t_1
            d_t = t_2 - t_1

            # CHANGED: pass atom_types instead of dataset_idx + spacegroup
            with torch.no_grad():
                pred_x_1 = model(x, t, atom_types, token_mask, x_sc)

            clean_traj.append(pred_x_1)
            if self.self_condition:
                x_sc = pred_x_1

            x_t_2 = self._x_euler_step(d_t, t_1, pred_x_1, x_t_1)
            tokens_traj.append(x_t_2)
            t_1 = t_2

        # Final integration step
        t_1 = ts[-1]
        x_t_1 = tokens_traj[-1]
        x = x_t_1 if self.corrupt else x_1
        if not self.corrupt and x_1 is None:
            raise ValueError("Must provide x_1 if not corrupting.")
        t = torch.ones((batch_size, 1), device=self.device) * t_1
        with torch.no_grad():
            pred_x_1 = model(x, t, atom_types, token_mask, x_sc)
        clean_traj.append(pred_x_1)
        tokens_traj.append(pred_x_1)

        return {"tokens_traj": tokens_traj, "clean_traj": clean_traj}

    def sample_with_classifier_free_guidance(
        self,
        batch_size,
        num_tokens,
        emb_dim,
        model,
        atom_types,
        cfg_scale=1.0,
        num_timesteps=None,
        x_0=None,
        x_1=None,
        token_mask=None,
        token_idx=None,
    ):
        """Generates new samples using the denoiser with classifier-free guidance.

        CHANGED: replaced dataset_idx + spacegroup with atom_types (B, N).
        Reason: CFG now operates over atom type conditioning instead of a global
        dataset label. The null condition is the special null token index
        (max_num_elements) defined in AtomTypeEmbedder, not dataset_idx=0.

        CFG doubles the batch: first half is conditional (real atom types),
        second half is unconditional (null atom type tokens everywhere).
        The final prediction is: uncond + cfg_scale * (cond - uncond).

        If cfg_scale=1.0 (default), this reduces to pure conditional sampling
        with no guidance amplification — a safe default for CSP since atom types
        are already a hard constraint, not a soft preference.

        Args:
            batch_size (int): Number of samples to generate: B.
            num_tokens (int): Max tokens per sample: N.
            emb_dim (int): Latent dimension: d.
            model (nn.Module): CrystalCSPDiT denoiser.
            atom_types (torch.Tensor): Atomic numbers (B, N). 0 for padding. CONDITION.
            cfg_scale (float): Guidance strength. 1.0 = no extra amplification.
            num_timesteps (int): ODE integration steps.
            x_0, x_1, token_mask, token_idx: same as sample().

        Returns:
            Dict with keys: tokens_traj, clean_traj.
        """
        if x_0 is None:
            x_0 = self._centered_gaussian(batch_size, num_tokens, emb_dim)
        if token_mask is None:
            token_mask = torch.ones(batch_size, num_tokens, device=self.device).bool()
        if token_idx is None:
            token_idx = torch.arange(num_tokens, device=self.device, dtype=torch.float32)[
                None
            ].repeat(batch_size, 1)

        # CHANGED: build null atom_types using max_num_elements as the null token index.
        # Original used dataset_idx=0 as the null class.
        # Now: null condition = every atom token replaced with the null embedding index.
        # model.atom_type_embedder.max_num_elements is the null index (see AtomTypeEmbedder).
        null_index = model.atom_type_embedder.max_num_elements
        atom_types_null = torch.full_like(atom_types, null_index)

        # Double the batch: [conditional | unconditional]
        x_0 = torch.cat([x_0, x_0], dim=0)                                      # (2B, N, d)
        atom_types_cfg = torch.cat([atom_types, atom_types_null], dim=0)         # (2B, N)
        token_mask_cfg = torch.cat([token_mask, token_mask], dim=0)              # (2B, N)

        if num_timesteps is None:
            num_timesteps = self.num_timesteps
        ts = torch.linspace(self.min_t, 1.0, num_timesteps)
        t_1 = ts[0]

        tokens_traj = [x_0]
        clean_traj = []
        x_sc = None
        for t_2 in ts[1:]:
            x_t_1 = tokens_traj[-1]
            if self.corrupt:
                x = x_t_1
            else:
                if x_1 is None:
                    raise ValueError("Must provide x_1 if not corrupting.")
                x = torch.cat([x_1, x_1], dim=0)
            t = torch.ones((2 * batch_size, 1), device=self.device) * t_1
            d_t = t_2 - t_1

            # CHANGED: forward_with_cfg now takes atom_types instead of dataset_idx + spacegroup
            with torch.no_grad():
                pred_x_1 = model.forward_with_cfg(
                    x, t, atom_types_cfg, token_mask_cfg, cfg_scale, x_sc
                )

            # Only keep the conditional half for the trajectory
            clean_traj.append(pred_x_1.chunk(2, dim=0)[0])
            if self.self_condition:
                x_sc = pred_x_1

            x_t_2 = self._x_euler_step(d_t, t_1, pred_x_1, x_t_1)
            tokens_traj.append(x_t_2)
            t_1 = t_2

        # Final integration step
        t_1 = ts[-1]
        x_t_1 = tokens_traj[-1]
        x = x_t_1 if self.corrupt else torch.cat([x_1, x_1], dim=0)
        if not self.corrupt and x_1 is None:
            raise ValueError("Must provide x_1 if not corrupting.")
        t = torch.ones((2 * batch_size, 1), device=self.device) * t_1
        with torch.no_grad():
            pred_x_1 = model.forward_with_cfg(
                x, t, atom_types_cfg, token_mask_cfg, cfg_scale, x_sc
            )
        clean_traj.append(pred_x_1.chunk(2, dim=0)[0])
        tokens_traj.append(pred_x_1)

        return {"tokens_traj": tokens_traj, "clean_traj": clean_traj}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_timesteps={self.num_timesteps}, self_condition={self.self_condition})"