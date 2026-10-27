"""Shared visual token encoder.

Architecture is unchanged from ``MEC-V8.2/encoder.py`` (commit 57e0f9b):
input [B,3,336,336] -> 24x24 patches -> 2 Transformer layers -> [B,576,384].
The CityNav adaptation only changes how the 336x336 RGB frame is produced.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.config import EncoderConfig


class VisualTokenEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = EncoderConfig.image_size,
        patch_size: int = EncoderConfig.patch_size,
        token_dim: int = EncoderConfig.token_dim,
        num_layers: int = EncoderConfig.num_layers,
        num_heads: int = EncoderConfig.num_heads,
        mlp_ratio: int = EncoderConfig.mlp_ratio,
        dropout: float = EncoderConfig.dropout,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.token_dim = int(token_dim)
        self.grid_size = self.image_size // self.patch_size
        self.num_tokens = self.grid_size ** 2

        self.patch_embed = nn.Conv2d(
            3, token_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, token_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * int(mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected [B,3,H,W], got {tuple(x.shape)}")
        z = self.patch_embed(x)
        z = z.flatten(2).transpose(1, 2)
        z = z + self.pos_embed
        z = self.encoder(z)
        return self.output_norm(z)


def load_encoder_checkpoint(
    model: VisualTokenEncoder,
    path: Path | str,
    *,
    map_location: str = "cpu",
) -> VisualTokenEncoder:
    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
