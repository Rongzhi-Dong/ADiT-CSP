"""
Copyright (c) Meta Platforms, Inc. and affiliates.

MODIFIED FOR CRYSTAL STRUCTURE PREDICTION (CSP):
    - Removed: 3D Cartesian coordinate (pos) embedder
      Reason: For crystals, pos is redundant — it is fully determined by frac_coords and the
      lattice (X = L @ F). The original code zeroed out the pos loss for crystals anyway.

    - Changed: atom_type_embedder is now a CONDITIONING input, not a reconstruction target.
      Reason: In CSP, atom types are *given* (known composition). The encoder receives them as
      context so the latent z only needs to encode structural geometry (frac coords + lattice).
      The decoder will also receive atom types explicitly, so the VAE never has to predict them.

    - Added: explicit lattice embedder broadcast to all atom tokens.
      Reason: The original code encoded lattice implicitly through frac coords.
      Making it explicit gives the encoder a direct signal about cell shape, which helps
      the decoder reconstruct accurate lengths/angles independently from frac coords.
"""

import math
from typing import Dict

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch


def get_index_embedding(indices, emb_dim, max_len=2048):
    """Creates sine/cosine positional embeddings from prespecified indices.

    Args:
        indices: offsets of size [..., num_tokens] of type integer
        emb_dim: dimension of the embeddings to create
        max_len: maximum sequence length

    Returns:
        positional embedding of shape [..., num_tokens, emb_dim]
    """
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class CrystalCSPEncoder(nn.Module):
    """Conditional Transformer encoder for Crystal Structure Prediction (CSP).

    Encodes fractional coordinates and lattice parameters into a latent representation,
    conditioned on known atom types. The latent z therefore captures only structural
    geometry, not composition.

    Args:
        max_num_elements: Maximum atomic number in the dataset (for atom type embedding table)
        d_model: Hidden dimension of the Transformer
        nhead: Number of attention heads
        dim_feedforward: Feedforward dimension in each Transformer layer
        activation: Activation function ('gelu' or 'relu')
        dropout: Dropout rate
        norm_first: Whether to use pre-norm (recommended True for stability)
        bias: Whether linear layers use bias
        num_layers: Number of Transformer encoder layers
    """

    def __init__(
        self,
        max_num_elements: int = 100,
        d_model: int = 1024,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_first: bool = True,
        bias: bool = True,
        num_layers: int = 6,
    ):
        super().__init__()

        self.max_num_elements = max_num_elements
        self.d_model = d_model
        self.num_layers = num_layers

        # [CHANGE] atom_type_embedder is now a CONDITIONING signal.
        # Original: atom types were part of the input to be reconstructed.
        # New: atom types are given (CSP), so we embed them as context for the encoder.
        # The encoder uses this to understand *what* atoms are at each site,
        # which helps it encode the structure of those atoms accurately.
        self.atom_type_embedder = nn.Embedding(max_num_elements, d_model)

        # [KEEP] Fractional coordinates embedder — primary geometric input for crystals.
        self.frac_coords_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # [NEW] Explicit lattice embedder broadcast to all atom tokens.
        # Original: lattice was only encoded implicitly through frac coords.
        # New: we give the encoder a direct signal for cell shape (6 params: 3 lengths + 3 angles).
        # Input is the raw lattice parameters [lengths | angles] concatenated = (6,).
        self.lattice_embedder = nn.Sequential(
            nn.Linear(6, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # [REMOVED] pos_embedder (3D Cartesian coordinates).
        # Reason: for crystals, pos = cell @ frac_coords, so it's redundant.
        # Keeping it would add noise without information gain.

        activation_fn = {
            "gelu": nn.GELU(approximate="tanh"),
            "relu": nn.ReLU(),
        }[activation]

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation_fn,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
                bias=bias,
            ),
            norm=nn.LayerNorm(d_model),
            num_layers=num_layers,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: PyG Data object with:
                atom_types    (N_total,)       — atomic numbers, used as CONDITION
                frac_coords   (N_total, 3)     — fractional coordinates [0,1)
                lengths       (B, 3)           — lattice lengths (Angstrom)
                angles_radians(B, 3)           — lattice angles (radians)
                num_atoms     (B,)             — number of atoms per crystal
                batch         (N_total,)       — batch index per atom
                token_idx     (N_total,)       — token index within each crystal

        Returns:
            dict with keys: x, num_atoms, batch, token_idx
        """
        # --- Condition: atom type embedding (given, not reconstructed) ---
        # Shape: (N_total, d_model)
        h = self.atom_type_embedder(batch.atom_types)

        # --- Input: fractional coordinates ---
        # Shape: (N_total, d_model)
        h = h + self.frac_coords_embedder(batch.frac_coords)

        # --- Input: lattice parameters (broadcast from crystal-level to atom-level) ---
        # Concatenate lengths and angles into a 6-dim vector per crystal: (B, 6)
        # Then index by batch to get per-atom lattice context: (N_total, 6)
        # [NEW] This makes cell shape directly available to every atom's token.
        lattice_params = torch.cat(
            [batch.lengths_scaled, batch.angles_radians], dim=-1
        )  # (B, 6)
        lattice_per_atom = lattice_params[batch.batch]  # (N_total, 6)
        h = h + self.lattice_embedder(lattice_per_atom)

        # --- Positional embedding (token order within each crystal) ---
        h = h + get_index_embedding(batch.token_idx, self.d_model)

        # --- Transformer: convert sparse PyG format to dense padded batch ---
        x, token_mask = to_dense_batch(h, batch.batch)  # (B, N_max, d_model)
        x = self.transformer(x, src_key_padding_mask=(~token_mask))
        x = x[token_mask]  # back to sparse (N_total, d_model)

        return {
            "x": x,
            "num_atoms": batch.num_atoms,
            "batch": batch.batch,
            "token_idx": batch.token_idx,
            # [NEW] Pass atom_types through so the decoder can use them as conditioning
            "atom_types": batch.atom_types,
        }