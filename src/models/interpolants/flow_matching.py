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

    # def corrupt_batch(self, batch):
    #     """Corrupts a batch of data by sampling a time t and interpolating to noisy samples.

    #     Args:
    #         batch (dict): Batch of clean data with keys:
    #             - x_1 (torch.Tensor): Clean data tensor.
    #             - token_mask (torch.Tensor): True if valid token, False if padding.
    #             - diffuse_mask (torch.Tensor): True if diffusion is to be performed, False if fixed during denoising.
    #     """
    #     noisy_batch = copy.deepcopy(batch)

    #     # [B, N, d]
    #     x_1 = batch["x_1"]

    #     # [B, N]
    #     token_mask = batch["token_mask"]
    #     diffuse_mask = batch["diffuse_mask"]
    #     batch_size, _ = diffuse_mask.shape

    #     # [B, 1]
    #     t = self._sample_t(batch_size)[:, None]
    #     noisy_batch["t"] = t

    #     # Apply corruptions
    #     if self.corrupt:
    #         x_t = self._corrupt_x(x_1, t, token_mask, diffuse_mask)
    #     else:
    #         x_t = x_1
    #     if torch.any(torch.isnan(x_t)):
    #         raise ValueError("NaN in x_t during corruption")
    #     noisy_batch["x_t"] = x_t

    #     return noisy_batch
    
    ### modified by RZD
    # BEFORE: original code probably noised entire x_1, e.g.
    # eps = torch.randn_like(x_1)
    # x_t = q_sample(x_1, eps, t)

    # AFTER: noise only continuous channels, keep channel 0 fixed (type channel)
    def corrupt_batch(self, dense_encoded_batch):
        """
        dense_encoded_batch expected fields:
        - x_1 : (B, N, D_total)  (D_total = d_cont + 1)
        - token_mask / diffuse_mask, etc.
        - atom_types : (B, N)  (optional, ints)
        """
        # noisy_batch = copy.deepcopy(dense_encoded_batch)
        # [B, N, d]
        # x_1 = batch["x_1"]
        # [B, N]
        token_mask = dense_encoded_batch["token_mask"]
        diffuse_mask = dense_encoded_batch["diffuse_mask"]
        batch_size, _ = diffuse_mask.shape

        x_1 = dense_encoded_batch["x_1"]  # (B,N,D_total)
        t = self._sample_t(batch_size)[:, None] # [B, 1]

        # split
        z_type = x_1[..., :1]    # (B,N,1) deterministic
        z_cont0 = x_1[..., 1:]   # (B,N,d_cont) to be corrupted

        if self.corrupt:
            x_cont_t = self._corrupt_x(z_cont0, t, token_mask, diffuse_mask)  # implement using existing schedule functions
            x_t = torch.cat([z_type, x_cont_t], dim=-1)  # (B,N,D_total)
        else:
            x_t = x_1
        if torch.any(torch.isnan(x_t)):
            raise ValueError("NaN in x_t during corruption")
      

        # Put back into dense_encoded_batch (keep other fields intact)
        noisy_dense_encoded_batch = dict(dense_encoded_batch)  # shallow copy
        noisy_dense_encoded_batch["x_t"] = x_t
        noisy_dense_encoded_batch["t"] = t
        # keep token_mask, diffuse_mask etc.
        return noisy_dense_encoded_batch
    ### END

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
        dataset_idx,
        spacegroup,
        num_timesteps=None,
        x_0=None,
        x_1=None,
        token_mask=None,
        token_idx=None,
    ):
        """Generates new samples of a specified (B, N, d) using denoiser model.

        Args:
            batch_size (int): Number of samples to generate.
            num_tokens (int): Number of tokens in each sample.
            emb_dim (int): Dimension of each token.
            model (nn.Module): Denoiser model to use.
            dataset_idx (torch.Tensor): Dataset index, used for classifier-free guidance. (B, 1)
            spacegroup (torch.Tensor): Spacegroup, used for classifier-free guidance. (B, 1)
            num_timesteps (int): Number of timesteps to integrate over.
            x_0 (torch.Tensor): Initial sample to start from.
            x_1 (torch.Tensor): Final sample to end at.
            token_mask (torch.Tensor): Mask for valid tokens.
            token_idx (torch.Tensor): Index of each token.

        Returns:
            Dict with keys:
                tokens_traj (list): List of generated samples at each timestep.
                clean_traj (list): List of denoised samples at each timestep.
        """
        # Set-up initial prior samples
        if x_0 is None:
            x_0 = self._centered_gaussian(batch_size, num_tokens, emb_dim)
        if token_mask is None:
            token_mask = torch.ones(batch_size, num_tokens, device=self.device).bool()
        if token_idx is None:
            token_idx = torch.arange(num_tokens, device=self.device, dtype=torch.float32)[
                None
            ].repeat(batch_size, 1)

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self.num_timesteps
        ts = torch.linspace(self.min_t, 1.0, num_timesteps)
        t_1 = ts[0]

        tokens_traj = [x_0]
        clean_traj = []
        x_sc = None
        for t_2 in ts[1:]:
            # Run denoiser model
            x_t_1 = tokens_traj[-1]
            if self.corrupt:
                x = x_t_1
            else:
                if x_1 is None:
                    raise ValueError("Must provide x_1 if not corrupting.")
                x = x_1
            t = torch.ones((batch_size, 1), device=self.device) * t_1
            d_t = t_2 - t_1

            # Run denoiser model
            with torch.no_grad():
                pred_x_1 = model(x, t, dataset_idx, spacegroup, token_mask, x_sc)

            # Process model output
            clean_traj.append(pred_x_1)
            if self.self_condition:
                x_sc = pred_x_1

            # Take reverse step
            x_t_2 = self._x_euler_step(d_t, t_1, pred_x_1, x_t_1)

            tokens_traj.append(x_t_2)
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        x_t_1 = tokens_traj[-1]
        if self.corrupt:
            x = x_t_1
        else:
            if x_1 is None:
                raise ValueError("Must provide x_1 if not corrupting.")
            x = x_1
        t = torch.ones((batch_size, 1), device=self.device) * t_1
        with torch.no_grad():
            pred_x_1 = model(x, t, dataset_idx, spacegroup, token_mask, x_sc)
        clean_traj.append(pred_x_1)
        tokens_traj.append(pred_x_1)

        return {"tokens_traj": tokens_traj, "clean_traj": clean_traj}

    def sample_with_classifier_free_guidance(
        self,
        batch_size,
        num_tokens,
        emb_dim,
        model,
        dataset_idx,
        spacegroup,
        cfg_scale=4.0,
        num_timesteps=None,
        x_0=None,
        x_1=None,
        token_mask=None,
        token_idx=None,
        atom_types=None,
    ):
        """Generates new samples of a specified (B, N, d) using denoiser model with classifier-free
        guidance.

        To be used with DiT denoisers, which use a different forward pass signature.

        Args:
            batch_size (int): Number of samples to generate: B.
            num_tokens (int): Max number of tokens in each sample: N.
            emb_dim (int): Dimension of each token: d.
            model (nn.Module): Denoiser model to use.
            dataset_idx (torch.Tensor): Dataset index, used for classifier-free guidance. (B, 1)
            spacegroup (torch.Tensor): Spacegroup, used for classifier-free guidance. (B, 1)
            cfg_scale (float): Scale factor for classifier-free guidance.
            num_timesteps (int): Number of timesteps to integrate over.
            x_0 (torch.Tensor): Initial sample to start from. (B, N, d)
            x_1 (torch.Tensor): Final sample to end at. (B, N, d)
            token_mask (torch.Tensor): Mask for valid tokens. (B, N)
            token_idx (torch.Tensor): Index of each token.

        Returns:
            Dict with keys:
                tokens_traj (list): List of generated samples at each timestep.
                clean_traj (list): List of denoised samples at each timestep.
        """
        assert torch.all(dataset_idx > 0), "Dataset 0 -> null class"

        # Set-up initial prior samples
        # if x_0 is None:
        #     x_0 = self._centered_gaussian(batch_size, num_tokens, emb_dim)
        
        ### modified by RZD
        # Set-up initial prior samples
        # If caller supplies `atom_types`, build x_0 so channel-0 == type and channels 1: == gaussian noise.
        if x_0 is None:
            if atom_types is None:
                x_0 = self._centered_gaussian(batch_size, num_tokens, emb_dim)
            else:
                # atom_types is now the encoded z_type [B, N, 1]
                z_type = atom_types if atom_types.dim() == 3 else atom_types.unsqueeze(-1)
                d_cont = int(emb_dim) - 1
                z_cont = self._centered_gaussian(batch_size, num_tokens, d_cont)
                x_0 = torch.cat([z_type.to(z_cont.dtype), z_cont], dim=-1) #
        
        
        # if x_0 is None:
        #     if atom_types is None:
        #         x_0 = self._centered_gaussian(batch_size, num_tokens, emb_dim)
        #     else:
        #         # build deterministic type channel from atom_types: shape (B, N, 1)
        #         if atom_types.dim() == 2:
        #             z_type = atom_types.to(device=self.device, dtype=torch.float32).unsqueeze(-1)
        #         else:
        #             z_type = atom_types.to(device=self.device, dtype=torch.float32)
        #             if z_type.shape[-1] != 1:
        #                 z_type = z_type[..., :1]
        #         # continuous part random
        #         d_cont = int(emb_dim) - 1
        #         if d_cont < 0:
        #             raise ValueError(f"emb_dim must be >=1, got {emb_dim}")
        #         z_cont = self._centered_gaussian(batch_size, num_tokens, d_cont)
        #         x_0 = torch.cat([z_type.to(z_cont.dtype), z_cont], dim=-1)
        ### end
                
        if token_mask is None:
            token_mask = torch.ones(batch_size, num_tokens, device=self.device).bool()
        if token_idx is None:
            token_idx = torch.arange(num_tokens, device=self.device, dtype=torch.float32)[
                None
            ].repeat(batch_size, 1)

        # Set-up classifier-free guidance
        x_0 = torch.cat([x_0, x_0], dim=0)  # (2B, N, d)
        dataset_idx_null = torch.zeros_like(dataset_idx)
        dataset_idx = torch.cat([dataset_idx, dataset_idx_null], dim=0)  # (2B, 1)
        spacegroup_null = torch.zeros_like(spacegroup)
        spacegroup = torch.cat([spacegroup, spacegroup_null], dim=0)  # (2B, 1)
        token_mask = torch.cat([token_mask, token_mask], dim=0)  # (2B, N)

        ### modified by RZD
        # If atom_types was provided, build its doubled version to enforce channel-0 during sampling.
        if atom_types is not None:
            # atom_types: (B,N) or (B,N,1) -> make (2B,N,1)
            if atom_types.dim() == 2:
                z_type = atom_types.to(device=self.device, dtype=torch.float32).unsqueeze(-1)
            else:
                z_type = atom_types.to(device=self.device, dtype=torch.float32)
                if z_type.shape[-1] != 1:
                    z_type = z_type[..., :1]
            z_type_doubled = torch.cat([z_type, z_type], dim=0)  # (2B, N, 1)
        else:
            z_type_doubled = None
        ### END


        # Set-up time
        if num_timesteps is None:
            num_timesteps = self.num_timesteps
        ts = torch.linspace(self.min_t, 1.0, num_timesteps)
        t_1 = ts[0]

        tokens_traj = [x_0]
        clean_traj = []
        x_sc = None
        
        for t_2 in ts[1:]:
            # Set-up input to denoiser
            x_t_1 = tokens_traj[-1]
            if self.corrupt:
                x = x_t_1
            else:
                if x_1 is None:
                    raise ValueError("Must provide x_1 if not corrupting.")
                x = torch.cat([x_1, x_1], dim=0)  # (2B, N, d)
            t = torch.ones((2 * batch_size, 1), device=self.device) * t_1
            d_t = t_2 - t_1

            # Run denoiser model
            with torch.no_grad():
                pred_x_1 = model.forward_with_cfg(
                    x, t, dataset_idx, spacegroup, token_mask, cfg_scale, x_sc
                )


            ### modified by RZD
            # If caller provided atom_types, enforce the first channel of pred to z_type (prevent drift)
            # if z_type_doubled is not None:
            #     # pred_x_1 shape: (2B, N, d); ensure dtype/device match
            #     pred_x_1 = pred_x_1.clone()
            #     pred_x_1[..., 0:1] = z_type_doubled.to(pred_x_1.dtype)
            
            ### END
            
            # Process model output          
            clean_traj.append(pred_x_1.chunk(2, dim=0)[0])  # Remove null class samples
            if self.self_condition:
                x_sc = pred_x_1

            # Take reverse step
            x_t_2 = self._x_euler_step(d_t, t_1, pred_x_1, x_t_1)
            ### added by RZD pn Feb 9 #####
            # if z_type_doubled is not None:
            #     x_t_2[..., 0:1] = z_type_doubled.to(x_t_2.dtype)
            ###### end ######
            tokens_traj.append(x_t_2)
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        x_t_1 = tokens_traj[-1]
        if self.corrupt:
            x = x_t_1
        else:
            if x_1 is None:
                raise ValueError("Must provide x_1 if not corrupting.")
            x = x_1
        t = torch.ones((2 * batch_size, 1), device=self.device) * t_1
        with torch.no_grad():
            pred_x_1 = model.forward_with_cfg(
                x, t, dataset_idx, spacegroup, token_mask, cfg_scale, x_sc
            )
        ### mdified by RZD
        if z_type_doubled is not None:
            pred_x_1 = pred_x_1.clone()
            pred_x_1[..., 0:1] = z_type_doubled.to(pred_x_1.dtype)
        ### END
        clean_traj.append(pred_x_1.chunk(2, dim=0)[0])  # Remove null class samples
        ### added by RZD on Feb 9 #####       
        if z_type_doubled is not None:
            pred_x_1[..., 0:1] = z_type_doubled.to(pred_x_1.dtype)
        #### end #######
        tokens_traj.append(pred_x_1)

        return {"tokens_traj": tokens_traj, "clean_traj": clean_traj}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_timesteps={self.num_timesteps}, self_condition={self.self_condition})"
