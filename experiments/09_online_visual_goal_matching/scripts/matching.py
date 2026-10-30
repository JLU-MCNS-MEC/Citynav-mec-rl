#!/usr/bin/env python3
"""Frozen-encoder visual matching for Stage 09.

All matching happens in the north-up frame: the heading observation is rotated
to north-up with ``geom.to_northup`` (verified on 0/90/180/270 deg) and the goal
image is rescaled to the same metres-per-pixel (``geom.goal_to_current_scale``).
The goal centre is therefore the target point, and the observation centre is
the UAV.

Three matchers share the same inputs:

* ``rgb_ncc``   - TM_CCOEFF_NORMED template matching (pixel baseline);
* ``layer3``    - 3x3-cell dense cosine matching on layer3 (14x14);
* ``layer2``    - dense cosine matching on layer2 (28x28).

Only the frozen ResNet is used; nothing is trained.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geom  # noqa: E402

EPS = 1e-8
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class Encoder:
    """Frozen ResNet-50 exposing layer2 / layer3 / layer4 feature maps."""

    def __init__(self, device: str = "cuda"):
        import torch
        import torchvision

        model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.torch = torch
        self.device = device
        self.stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                                        model.layer1, model.layer2).to(device).eval()
        self.layer3 = model.layer3.to(device).eval()
        self.layer4 = model.layer4.to(device).eval()
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def features(self, images: np.ndarray, layers: tuple[str, ...] = ("layer2", "layer3")) -> dict:
        """images: (B,H,W,3) uint8 -> dict of float32 (B,C,H,W) numpy arrays."""
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(images)).to(self.device)
        x = x.permute(0, 3, 1, 2).float().div_(255.0).sub_(self.mean).div_(self.std)
        with torch.no_grad():
            f2 = self.stem(x)
            out = {"layer2": f2}
            if "layer3" in layers or "layer4" in layers:
                f3 = self.layer3(f2)
                if "layer3" in layers:
                    out["layer3"] = f3
                if "layer4" in layers:
                    out["layer4"] = self.layer4(f3)
        return {k: v.float().cpu().numpy() for k, v in out.items()}


def prepare_observation(raster, x: float, y: float, yaw: float, half: float = geom.CUR_HALF_M):
    img = raster.heading_crop(float(x), float(y), float(yaw), half=half)
    return geom.to_northup(img, yaw)


def prepare_goal(raster, tx: float, ty: float):
    img = raster.north_crop(float(tx), float(ty), geom.GOAL_HALF_M)
    return geom.goal_to_current_scale(img)


def _suppress_margin(score: np.ndarray, peak: tuple[int, int], radius: int) -> float:
    """top1 - best score outside a (2r+1) window around the peak."""
    r0, c0 = peak
    mask = np.ones_like(score, bool)
    mask[max(r0 - radius, 0):r0 + radius + 1, max(c0 - radius, 0):c0 + radius + 1] = False
    second = float(score[mask].max()) if mask.any() else 0.0
    return float(score[r0, c0]) - second


def match_rgb(obs: np.ndarray, goal: np.ndarray, template_px: int = 96):
    """TM_CCOEFF_NORMED template match; returns (col,row,score,margin)."""
    c = goal.shape[0] // 2
    half = template_px // 2
    tpl = goal[c - half:c + half, c - half:c + half]
    if tpl.std() < 1e-6:
        return None
    score = cv2.matchTemplate(obs, tpl, cv2.TM_CCOEFF_NORMED)
    _, _, _, peak = cv2.minMaxLoc(score)
    sc = float(score[peak[1], peak[0]])
    margin = _suppress_margin(score, (peak[1], peak[0]), template_px // 2)
    col = peak[0] + half - 0.5
    row = peak[1] + half - 0.5
    return col, row, sc, margin


def _dense_descriptor(feat: np.ndarray, block: int) -> tuple[np.ndarray, int, int]:
    """Central block of a (C,H,W) map -> (C,) mean descriptor, plus its centre."""
    c, h, w = feat.shape
    ci, cj = (h - 1) / 2.0, (w - 1) / 2.0
    r0, r1 = int(math.floor(ci - block / 2 + 0.5)), int(math.ceil(ci + block / 2 - 0.5)) + 1
    c0, c1 = int(math.floor(cj - block / 2 + 0.5)), int(math.ceil(cj + block / 2 - 0.5)) + 1
    r0, r1 = max(r0, 0), min(r1, h)
    c0, c1 = max(c0, 0), min(c1, w)
    desc = feat[:, r0:r1, c0:c1].mean(axis=(1, 2))
    return desc / (np.linalg.norm(desc) + EPS), r0, c0


def match_dense(obs_feat: np.ndarray, goal_feat: np.ndarray, block: int = 2,
                out_px: int = geom.OUT):
    """Cosine matching of the goal centre block against every observation cell."""
    desc, _, _ = _dense_descriptor(goal_feat, block)
    c, h, w = obs_feat.shape
    from scipy.ndimage import uniform_filter
    pooled = uniform_filter(obs_feat.astype(np.float32), size=(1, block, block),
                            mode="nearest")
    norm = np.linalg.norm(pooled, axis=0) + EPS
    score = np.tensordot(desc, pooled, axes=([0], [0])) / norm
    idx = np.unravel_index(int(np.argmax(score)), score.shape)
    sc = float(score[idx])
    margin = _suppress_margin(score, idx, 1)
    cell = out_px / float(h)
    col = (idx[1] + 0.5) * cell
    row = (idx[0] + 0.5) * cell
    return col, row, sc, margin


def pixel_to_offset(col: float, row: float, scale: float = geom.CUR_SCALE,
                    out: int = geom.OUT) -> tuple[float, float]:
    """North-up pixel -> world offset (dx, dy) from the UAV."""
    c = geom.center_px(out)
    return (col - c) * scale, -(row - c) * scale


def offset_to_pixel(dx: float, dy: float, scale: float = geom.CUR_SCALE,
                    out: int = geom.OUT) -> tuple[float, float]:
    c = geom.center_px(out)
    return c + dx / scale, c - dy / scale


def localisation_error(match_col: float, match_row: float,
                       true_col: float, true_row: float,
                       scale: float = geom.CUR_SCALE) -> float:
    return math.hypot((match_col - true_col) * scale, (match_row - true_row) * scale)
