"""Frozen-encoder tokenization and heatmap construction.

This is the CityNav-side equivalent of ``MEC-V8.2/token_data.py``:

  current RGB 336x336 -> 576 x 384 tokens
  goal RGB S x 96x96 -> S x 384 tokens (foreground-weighted)
  heatmap = cosine(current tokens, goal tokens) pooled to 8x8

No simulator goal mask, coordinate, or GSM map is used.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.config import EncoderConfig
from src.models.mec_common import build_spatial_heatmap


ENCODER = EncoderConfig()


def images_to_tensor(
    images: np.ndarray,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    array = np.asarray(images, dtype=np.uint8)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"expected [N,H,W,3], got {array.shape}")
    tensor = torch.from_numpy(array).float().permute(0, 3, 1, 2) / 255.0
    return tensor.to(device)


@torch.no_grad()
def encode_current_tokens(
    encoder: torch.nn.Module,
    current_rgbs: np.ndarray,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    x = images_to_tensor(current_rgbs, device)
    if x.shape[-2:] != (ENCODER.image_size, ENCODER.image_size):
        x = F.interpolate(
            x,
            size=(ENCODER.image_size, ENCODER.image_size),
            mode="bilinear",
            align_corners=False,
        )
    return encoder(x)


@torch.no_grad()
def encode_goal_tokens(
    encoder: torch.nn.Module,
    goal_rgbs: np.ndarray,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return [B,S,384] foreground-weighted Goal reference tokens."""
    array = np.asarray(goal_rgbs, dtype=np.uint8)
    if array.ndim != 5 or array.shape[-1] != 3:
        raise ValueError(f"expected [B,S,H,W,3], got {array.shape}")
    b, s, h, w, _ = array.shape
    flat = array.reshape(b * s, h, w, 3)
    x = images_to_tensor(flat, device)
    if x.shape[-2:] != (ENCODER.image_size, ENCODER.image_size):
        x = F.interpolate(
            x,
            size=(ENCODER.image_size, ENCODER.image_size),
            mode="bilinear",
            align_corners=False,
        )
    full_tokens = encoder(x)
    full_tokens = full_tokens.reshape(b, s, ENCODER.num_tokens, ENCODER.token_dim)

    raw = torch.from_numpy(array).float().to(device) / 255.0
    background = raw[:, :, 0:1, 0:1, :]
    difference = (raw - background).abs().amax(dim=-1)
    foreground = (difference > (8.0 / 255.0)).float().unsqueeze(2)
    foreground = F.interpolate(
        foreground.reshape(b * s, 1, h, w),
        size=(ENCODER.grid_size, ENCODER.grid_size),
        mode="area",
    ).reshape(b, s, ENCODER.num_tokens)
    weights = foreground.to(device)
    denominator = weights.sum(dim=-1, keepdim=True)
    empty = denominator.squeeze(-1) < 1e-6
    if bool(empty.any()):
        weights = weights.clone()
        weights[empty] = 0.0
        center = (
            (ENCODER.grid_size // 2) * ENCODER.grid_size
            + ENCODER.grid_size // 2
        )
        weights[empty, :, center] = 1.0
        denominator = weights.sum(dim=-1, keepdim=True)
    return (full_tokens * weights.unsqueeze(-1)).sum(dim=2) / denominator


@torch.no_grad()
def encode_scene(
    encoder: torch.nn.Module,
    current_rgbs: np.ndarray,
    goal_rgbs: np.ndarray,
    device: str | torch.device = "cpu",
) -> dict:
    current_tokens = encode_current_tokens(encoder, current_rgbs, device)
    goal_tokens = encode_goal_tokens(encoder, goal_rgbs, device)
    heatmap = build_spatial_heatmap(current_tokens, goal_tokens)
    return {
        "current_tokens": current_tokens,
        "goal_tokens": goal_tokens,
        "heatmap": heatmap,
    }
