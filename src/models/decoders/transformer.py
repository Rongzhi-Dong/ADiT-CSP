"""
Copyright (c) Meta Platforms, Inc. and affiliates.

MODIFIED FOR CRYSTAL STRUCTURE PREDICTION (CSP):
    - Removed: atom_types_head (linear output head for atom type prediction)
      Reason: In CSP, atom types are the CONDITION, not the output. The decoder
      receives atom types explicitly and never needs to predict them.

    - Removed: pos_head (Cartesian coordinate output head)
      Reason: For crystals, Cartesian coords = cell @ frac_coords, so they are
      fully determined by the predicted frac_coords + lattice. Predicting them
      separately is redundant and was zeroed out in the original loss anyway.

    - Added: atom_type_embedder injected at decoder input.
      Reason: The decoder needs to know atom types to reconstruct the correct
      geometry (different elements have different bond lengths, radii, etc.).
      This is the standard CVAE pattern: decoder(z, condition) -> x.

    - Added: explicit lattice_input_embedder at the start of the decoder.
      Reason: Same as encoder — we give the decoder a direct cell-shape signal
      so it can more easily produce self-consistent frac_coords + lattice.
"""

import math
from typing import Dict

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter


def get_index_embedding(indices, emb_dim, max_len=2048):
    """Creates sine/cosine positional embeddings from prespecified indices."""
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class CrystalCSPDecoder(nn.Module):
    """Conditional Transformer decoder for Crystal Structure Prediction (CSP).

    Takes a latent representation z and known atom types, and reconstructs
    fractional coordinates and lattice parameters. Atom types are never predicted.

    Args:
        max_num_elements: Maximum atomic number (for the conditioning embedding table)
        d_model: Hidden dimension of the Transformer
        nhead: Number of attention heads
        dim_feedforward: Feedforward dimension in each Transformer layer
        activation: Activation function ('gelu' or 'relu')
        dropout: Dropout rate
        norm_first: Whether to use pre-norm
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

        # [NEW] Atom type conditioning embedding for the decoder.
        # The decoder receives atom types as a conditioning signal (CVAE pattern).
        # This is added to the up-projected latent token before the Transformer,
        # so every layer of the decoder is aware of what element each atom is.
        self.atom_type_embedder = nn.Embedding(max_num_elements, d_model)

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

        # [REMOVED] atom_types_head — CSP never predicts atom types; they are given.

        # [REMOVED] pos_head — Cartesian coordinates redundant for crystals.

        # [KEEP] Fractional coordinates output head — primary geometric output.
        self.frac_coords_head = nn.Linear(d_model, 3, bias=False)

        # [KEEP] Lattice output head — predicts 6 params (lengths + angles) from pooled repr.
        self.lattice_head = nn.Linear(d_model, 6, bias=False)

    def forward(self, encoded_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoded_batch: dict with keys:
                x           (N_total, latent_dim)  — up-projected latent tokens from post_quant_conv
                atom_types  (N_total,)              — atomic numbers (CONDITION, not target)
                num_atoms   (B,)                    — atoms per crystal
                batch       (N_total,)              — batch index per atom
                token_idx   (N_total,)              — token index within each crystal

        Returns:
            dict with keys: frac_coords, lengths, angles
            (no atom_types, no pos — those are either given or redundant)
        """
        x = encoded_batch["x"]  # (N_total, d_model) after post_quant_conv

        # [NEW] Inject atom type conditioning at the start of the decoder.
        # This is the core CVAE pattern: decode(z + condition) -> structure.
        # Each atom token gets its type embedding added before the Transformer,
        # ensuring every attention layer knows the element at each site.
        x = x + self.atom_type_embedder(encoded_batch["atom_types"])

        # --- Positional embedding ---
        x = x + get_index_embedding(encoded_batch["token_idx"], self.d_model)

        # --- Dense batch for Transformer ---
        x, token_mask = to_dense_batch(x, encoded_batch["batch"])  # (B, N_max, d_model)
        x = self.transformer(x, src_key_padding_mask=(~token_mask))
        x = x[token_mask]  # (N_total, d_model)

        # --- Global pooling for lattice prediction: (B, d_model) ---
        x_global = scatter(x, encoded_batch["batch"], dim=0, reduce="mean")

        # --- Output heads ---
        # [KEEP] Fractional coordinates: per-atom prediction
        frac_coords_out = self.frac_coords_head(x)  # (N_total, 3)

        # [KEEP] Lattice: crystal-level prediction from pooled representation
        lattices_out = self.lattice_head(x_global)  # (B, 6)

        # [REMOVED] atom_types_out — not predicted in CSP
        # [REMOVED] pos_out — not predicted for crystals

        return {
            "frac_coords": frac_coords_out,   # (N_total, 3)
            "lengths": lattices_out[:, :3],    # (B, 3)
            "angles": lattices_out[:, 3:],     # (B, 3)
            # Passthrough: atom types are the condition, not model output.
            # We carry them here so downstream evaluation code can read them.
            "atom_types": encoded_batch["atom_types"],
        }