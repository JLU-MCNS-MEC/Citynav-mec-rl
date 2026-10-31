#!/usr/bin/env python3
"""Continuous target belief for Stage 09C.

A Gaussian mixture over the top-K global candidates:

* component i has a continuous 2-D mean (metres), an isotropic variance and a
  weight; the prior weights are the Stage-09B candidate softmax;
* an accepted visual measurement z (a continuous world point from sub-pixel
  matching) updates every component with a Kalman step and re-weights it by the
  tempered Gaussian likelihood;
* the estimate is the **mean of the MAP component**, so a target between two
  24 m grid points keeps its sub-grid offset instead of snapping to a candidate.

No true target is used here; z comes only from the images.
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-12


def subpixel_match(obs_feat: np.ndarray, goal_feat: np.ndarray, block: int = 2,
                   out_px: int = 224) -> dict:
    """Dense cosine match with a sub-pixel 3x3 weighted centroid.

    Returns the peak position in observation pixels, the score, the margin
    outside the peak neighbourhood and the peak sharpness.
    """
    from scipy.ndimage import uniform_filter
    from matching import _dense_descriptor, _suppress_margin

    desc, _, _ = _dense_descriptor(goal_feat, block)
    c, h, w = obs_feat.shape
    pooled = uniform_filter(obs_feat.astype(np.float32), size=(1, block, block),
                            mode="nearest")
    norm = np.linalg.norm(pooled, axis=0) + EPS
    score = np.tensordot(desc, pooled, axes=([0], [0])) / norm
    idx = np.unravel_index(int(np.argmax(score)), score.shape)
    r0, c0 = idx
    r_lo, r_hi = max(r0 - 1, 0), min(r0 + 2, h)
    c_lo, c_hi = max(c0 - 1, 0), min(c0 + 2, w)
    win = score[r_lo:r_hi, c_lo:c_hi].astype(np.float64)
    wgt = np.clip(win - win.min(), 0, None) ** 2
    if wgt.sum() <= 0:
        dr = dc = 0.0
    else:
        rr, cc = np.mgrid[r_lo:r_hi, c_lo:c_hi]
        dr = float((wgt * (rr - r0)).sum() / wgt.sum())
        dc = float((wgt * (cc - c0)).sum() / wgt.sum())
    cell = out_px / float(h)
    col = (c0 + 0.5 + dc) * cell
    row = (r0 + 0.5 + dr) * cell
    margin = _suppress_margin(score, idx, 1)
    # sharpness: how fast the peak falls inside its 3x3 window (0..1)
    peak = float(score[r0, c0])
    spread = float(np.clip(peak - win.mean(), 0.0, 1.0))
    return {"col": col, "row": row, "score": float(peak), "margin": float(margin),
            "sharpness": spread, "cell": cell, "heatmap": score}


class ContinuousBelief:
    """Top-K Gaussian mixture with Kalman updates and tempered re-weighting."""

    def __init__(self, positions: np.ndarray, sims: np.ndarray, temperature: float,
                 k: int = 8, sigma_prior_m: float = 12.0):
        order = np.argsort(-sims)[:k]
        self.means = np.asarray(positions, float)[order].copy()
        self.vars = np.full(len(order), float(sigma_prior_m) ** 2)
        logits = np.asarray(sims, float)[order] / temperature
        logits -= logits.max()
        w = np.exp(logits)
        self.weights = w / w.sum()
        self.last_update = -99
        self.last_accept = None

    # -- queries -----------------------------------------------------------
    def entropy(self) -> float:
        w = self.weights
        return float(-(w * np.log(w + EPS)).sum() / math.log(len(w)))

    def map_index(self) -> int:
        return int(np.argmax(self.weights))

    def estimate(self) -> np.ndarray:
        return self.means[self.map_index()].copy()

    def best_mahalanobis(self, z: np.ndarray) -> float:
        d2 = ((self.means - z) ** 2).sum(axis=1)
        var = self.vars + 1e-6
        return float(np.sqrt((d2 / var).min()))

    # -- update ------------------------------------------------------------
    def update(self, z: np.ndarray, sigma_vis_m: float, eta: float, rho: float,
               step: int, merge_m: float = 10.0, prune_w: float = 1e-4) -> None:
        var_z = float(sigma_vis_m) ** 2
        total = self.vars + var_z
        d2 = ((self.means - z) ** 2).sum(axis=1)
        ll = -0.5 * d2 / total - np.log(2 * math.pi * total)
        w_scale = eta * (rho if step - self.last_update <= 1 else 1.0)
        logits = np.log(self.weights + EPS) + w_scale * (ll - ll.max())
        logits -= logits.max()
        self.weights = np.exp(logits)
        self.weights /= self.weights.sum()
        # Kalman mean / variance update (full weight, tempered only for weights)
        gain = self.vars / total
        self.means = self.means + gain[:, None] * (z - self.means)
        self.vars = self.vars * (1.0 - gain)
        self.last_update = step
        self.last_accept = z.copy()
        self._merge(merge_m, prune_w)

    def _merge(self, merge_m: float, prune_w: float) -> None:
        keep = self.weights >= prune_w
        if keep.sum() == 0:
            keep[int(np.argmax(self.weights))] = True
        self.means, self.vars, self.weights = self.means[keep], self.vars[keep], self.weights[keep]
        self.weights /= self.weights.sum()
        used = np.zeros(len(self.weights), bool)
        for i in range(len(self.weights)):
            if used[i]:
                continue
            near = (~used) & (np.linalg.norm(self.means - self.means[i], axis=1) <= merge_m)
            near[i] = True
            w = self.weights[near]
            self.means[i] = (self.means[near] * w[:, None]).sum(axis=0) / w.sum()
            self.vars[i] = float((self.vars[near] * w).sum() / w.sum())
            self.weights[i] = float(w.sum())
            used |= near
        self.means, self.vars, self.weights = self.means[used], self.vars[used], self.weights[used]
        self.weights /= self.weights.sum()
