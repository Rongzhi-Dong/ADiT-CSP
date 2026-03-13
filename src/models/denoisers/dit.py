"""
Copyright (c) Meta Platforms, Inc. and affiliates.

MODIFIED FOR CRYSTAL STRUCTURE PREDICTION (CSP):

Summary of changes vs original dit.py:
    1. Added: AtomTypeEmbedder — a per-token atom type conditioning module.
       Reason: In CSP, atom types are known. The best way to inject them into
       the DiT is per-atom (token-level), since the latent z has shape (B, N, d)
       with a one-to-one correspondence between token i and atom i.
       This is more expressive than global conditioning via adaLN (which averages
       composition info into a single vector) while being simpler than cross-attention.

    2. Changed: DiT.forward() accepts atom_types tensor (B, N) and adds the
       atom type embedding to the noisy latent tokens before processing.
       Reason: This injects composition knowledge at every Transformer layer
       through the residual stream — the most direct way to condition on composition.

    3. Changed: DiT.forward_with_cfg() passes atom_types through.
       Reason: CFG still works the same way; atom type conditioning is
       deterministic (no dropout), so it applies equally to both conditional
       and unconditional predictions.

    4. Removed: spacegroup_embedder (optional, can be re-added later).
       Reason: Spacegroup conditioning is orthogonal to our changes. It can
       trivially be added back via adaLN alongside the timestep embedding.
       We remove it here for clarity, but the architecture supports it.

    5. Removed: dataset_embedder.
       Reason: Crystal-only model — no need to distinguish periodic vs.
       non-periodic. The binary class label from the original is no longer needed.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Embedding modules
# ─────────────────────────────────────────────────────────────────────────────

class TimestepEmbedder(nn.Module):
    """Embeds scalar diffusion timesteps into vector representations.
    Unchanged from original.
    """

    def __init__(self, hidden_dim, frequency_embedding_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_dim)
        return self.mlp(t_freq)


class AtomTypeEmbedder(nn.Module):
    """Per-token atom type conditioning for CSP.

    [NEW] Maps each atom's atomic number to a d_model-dimensional embedding.
    This is added directly to the noisy latent tokens so that every DiT layer
    is aware of the element type at each lattice site.

    Args:
        max_num_elements: Size of the embedding table (max atomic number + 1)
        hidden_dim: Dimension of the output embedding (= d_model of DiT)
        dropout_prob: If > 0, randomly drop atom type conditioning during training
                      to enable classifier-free guidance on composition.
                      Set to 0 if you always want to condition on atom types.
    """

    def __init__(self, max_num_elements: int, hidden_dim: int, dropout_prob: float = 0.0):
        super().__init__()
        # +1 for a null/mask token (index 0) used when dropout_prob > 0
        self.embedding_table = nn.Embedding(max_num_elements + 1, hidden_dim)
        self.dropout_prob = dropout_prob
        self.max_num_elements = max_num_elements

    def forward(self, atom_types: torch.Tensor, train: bool = True) -> torch.Tensor:
        """
        Args:
            atom_types: (B, N_max) — atomic numbers, 0 for padding tokens
            train: whether we are in training mode (enables dropout)
        Returns:
            embeddings: (B, N_max, hidden_dim)
        """
        if train and self.dropout_prob > 0:
            # Randomly zero out the atom type conditioning for full samples
            # (drop per-sample, not per-atom, to enable CFG on composition)
            drop_mask = torch.rand(atom_types.shape[0], 1, device=atom_types.device) < self.dropout_prob
            # Replace with null token (index = max_num_elements)
            null_tokens = torch.full_like(atom_types, self.max_num_elements)
            atom_types = torch.where(drop_mask.expand_as(atom_types), null_tokens, atom_types)
        return self.embedding_table(atom_types)


def get_pos_embedding(indices, emb_dim, max_len=2048):
    """Sine/cosine positional embeddings. Unchanged from original."""
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    return torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer blocks — unchanged from original
# ─────────────────────────────────────────────────────────────────────────────

class Mlp(nn.Module):
    """MLP block. Unchanged from original."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, norm_layer=None, bias=True, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning. Unchanged from original."""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=0, bias=True, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim,
                       act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, x, c, mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        _x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(_x, _x, _x, key_padding_mask=mask, need_weights=False)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Final DiT output layer. Unchanged from original."""

    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ─────────────────────────────────────────────────────────────────────────────
# Main DiT model for Crystal CSP
# ─────────────────────────────────────────────────────────────────────────────

class CrystalCSPDiT(nn.Module):
    """Diffusion Transformer for Crystal Structure Prediction.

    Differences from the original DiT:
        - atom_type_embedder (per-token) replaces dataset_embedder (global)
        - spacegroup_embedder removed (can be re-added as extra adaLN term)
        - forward() takes atom_types (B, N_max) as explicit argument
        - atom type embeddings are added to the noisy latent tokens (token-level conditioning)
        - timestep embedding still controls adaLN modulation (unchanged)

    Args:
        d_x: Latent dimension of VAE (input/output size of the DiT)
        d_model: Internal hidden dimension of the DiT
        num_layers: Number of DiT blocks
        nhead: Number of attention heads
        mlp_ratio: MLP expansion ratio in each block
        max_num_elements: Maximum atomic number (for atom type embedding table)
        atom_type_dropout_prob: Probability of dropping atom type conditioning
                                 during training (for CFG on composition; set 0 to disable)
    """

    def __init__(
        self,
        d_x: int = 8,
        d_model: int = 384,
        num_layers: int = 12,
        nhead: int = 6,
        mlp_ratio: float = 4.0,
        max_num_elements: int = 100,
        atom_type_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.d_x = d_x
        self.d_model = d_model
        self.nhead = nhead

        # Input projection: noisy latent + self-conditioning latent → d_model
        # Self-conditioning doubles the input width (same as original)
        self.x_embedder = nn.Linear(2 * d_x, d_model, bias=True)

        # Timestep conditioning via adaLN (unchanged from original)
        self.t_embedder = TimestepEmbedder(d_model)

        # [NEW] Per-token atom type conditioning.
        # The original used a global dataset label via adaLN.
        # We use per-atom embeddings added to the token stream instead,
        # which is more expressive and directly maps composition → structure.
        self.atom_type_embedder = AtomTypeEmbedder(
            max_num_elements=max_num_elements,
            hidden_dim=d_model,
            dropout_prob=atom_type_dropout_prob,
        )

        # [REMOVED] dataset_embedder — no periodic/non-periodic distinction needed
        # [REMOVED] spacegroup_embedder — can be re-added as: c = t + spacegroup_emb(sg)

        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, nhead, mlp_ratio=mlp_ratio) for _ in range(num_layers)]
        )
        self.final_layer = FinalLayer(d_model, d_x)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize atom type embedding table with small normal weights
        nn.init.normal_(self.atom_type_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation (standard DiT init for training stability)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        atom_types: torch.Tensor,
        mask: torch.Tensor,
        x_sc: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x           (B, N, d_x)  — noisy latent tokens at timestep t
            t           (B,)          — diffusion timestep per sample
            atom_types  (B, N)        — atomic numbers (0 for padding tokens); CONDITION
            mask        (B, N)        — True if valid token, False if padding
            x_sc        (B, N, d_x)  — self-conditioning prediction from previous step (optional)

        Returns:
            pred_x (B, N, d_x) — predicted clean latent tokens
        """
        # --- Positional embedding ---
        token_index = torch.cumsum(mask, dim=-1, dtype=torch.int64) - 1
        pos_emb = get_pos_embedding(token_index, self.d_model)  # (B, N, d_model)

        # --- Self-conditioning (same as original) ---
        if x_sc is None:
            x_sc = torch.zeros_like(x)
        # Project [x | x_sc] → d_model tokens
        h = self.x_embedder(torch.cat([x, x_sc], dim=-1)) + pos_emb  # (B, N, d_model)

        # [NEW] Add per-token atom type conditioning to the token stream.
        # This is the key change: composition is injected at the token level,
        # not as a global scalar via adaLN. Every DiT block sees atom identity
        # through the residual stream, giving fine-grained per-site conditioning.
        atom_type_emb = self.atom_type_embedder(atom_types, train=self.training)  # (B, N, d_model)
        h = h + atom_type_emb

        # --- Global conditioning for adaLN: only timestep ---
        # [CHANGED] Original: c = t_emb + dataset_emb + spacegroup_emb
        # New: c = t_emb only (atom types handled per-token above)
        c = self.t_embedder(t.squeeze(1))  # (B, d_model)

        # --- Transformer blocks ---
        for block in self.blocks:
            h = block(h, c, ~mask)  # (B, N, d_model)

        # --- Output projection ---
        pred_x = self.final_layer(h, c)  # (B, N, d_x)
        pred_x = pred_x * mask[..., None]  # zero out padding
        return pred_x

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        atom_types: torch.Tensor,
        mask: torch.Tensor,
        cfg_scale: float,
        x_sc: torch.Tensor = None,
    ) -> torch.Tensor:
        """Classifier-free guidance forward pass.

        [CHANGED] atom_types replaces dataset_idx as the conditioning signal.
        The first half of the batch is conditional (real atom types),
        the second half is unconditional (null atom types).

        The caller should prepare the batch as:
            atom_types = torch.cat([real_atom_types, null_atom_types], dim=0)
            x = torch.cat([x, x], dim=0)
        where null_atom_types uses index max_num_elements (the null embedding).
        """
        half = len(x) // 2
        model_out = self.forward(x, t, atom_types, mask, x_sc)
        cond_out, uncond_out = model_out[:half], model_out[half:]
        # Standard CFG interpolation
        guided_out = uncond_out + cfg_scale * (cond_out - uncond_out)
        return torch.cat([guided_out, guided_out], dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Standard model size configs (same as original)
# ─────────────────────────────────────────────────────────────────────────────

def CrystalCSPDiT_S(d_x=8, **kwargs):
    return CrystalCSPDiT(d_x=d_x, d_model=384, num_layers=12, nhead=6, **kwargs)

def CrystalCSPDiT_B(d_x=8, **kwargs):
    return CrystalCSPDiT(d_x=d_x, d_model=768, num_layers=12, nhead=12, **kwargs)

def CrystalCSPDiT_L(d_x=8, **kwargs):
    return CrystalCSPDiT(d_x=d_x, d_model=1024, num_layers=24, nhead=16, **kwargs)