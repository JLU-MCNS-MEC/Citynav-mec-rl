#!/usr/bin/env python3
"""Stage 10A learned matchers v2 -- FAIR variants built ON the C1 mechanism.

Fairness contract (the only parts that may differ from C1 are marked):

* goal descriptor:  IDENTICAL to C1 -- central 2x2 block of the goal layer2
                    map, mean-pooled, L2-normalised (``matching._dense_descriptor``);
* observation map: IDENTICAL to C1 -- 2x2-neighbourhood pooled layer2 cells
                    (``uniform_filter(size=(1,2,2), mode='nearest')``, replicated
                    exactly with replicate-pad + avg_pool2d on the same 28x28 grid);
* match map:        28x28, same grid and resolution as C1;
    - C1: cosine(desc, cell)                       [baseline, untouched]
    - PQ: cosine(MLP(desc), W_v . cell)            [LEARNABLE: query + value
                                                    projection; reduces to C1
                                                    for identity weights]
    - CA: softmax_c(q(desc) . k(cell) * scale)     [LEARNABLE: attention-weight
                                                    map replaces the cosine map]
* position:         IDENTICAL to C1 -- argmax peak + 3x3 weighted sub-pixel
                    centroid + margin suppression (faithful port of
                    ``continuous_belief.subpixel_match``);
* score:            map peak (same semantics as C1); gate tau re-calibrated on
                    the same 2,000-state Train-Seen subset;
* training:         map-level supervision only.  In-view: Gaussian blob
                    (sigma=1.5 cells) at the true cell; out-of-view: all-low.
                    PQ: MSE to the [-1,1] cosine map; CA: BCE on pre-softmax
                    logits.  Frozen ResNet-50 layer2, seed 42, no val tuning.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = 224.0
D = 512
EPS = 1e-8


# --------------------------------------------------------------------------
# C1-identical preprocessing (torch ports, verified against matching.py)
# --------------------------------------------------------------------------
def goal_desc(goal_feat: torch.Tensor) -> torch.Tensor:
    """Central 2x2 block of the 28x28 goal map, mean-pooled, L2-normalised.
    Exact torch port of ``matching._dense_descriptor(goal_feat, block=2)``."""
    h, w = goal_feat.shape[2], goal_feat.shape[3]
    ci = cj = (h - 1) / 2.0
    r0 = int(math.floor(ci - 2 / 2 + 0.5))
    r1 = int(math.ceil(ci + 2 / 2 - 0.5)) + 1
    c0 = int(math.floor(cj - 2 / 2 + 0.5))
    c1 = int(math.ceil(cj + 2 / 2 - 0.5)) + 1
    desc = goal_feat[:, :, r0:r1, c0:c1].mean(dim=(2, 3))
    return F.normalize(desc, dim=1, eps=EPS)


def obs_pooled(obs_feat: torch.Tensor) -> torch.Tensor:
    """2x2-neighbourhood mean pooling with edge replication.
    Exact torch port of ``uniform_filter(x, size=(1,2,2), mode='nearest')``:
    scipy's even-size window is asymmetric (mean over x[{i-1,i}, {j-1,j}] with
    the border replicated) -> separable: horizontal pair mean, then vertical
    pair mean, each with replicate padding on the left/top edge."""
    x = obs_feat
    left = F.pad(x[:, :, :, :-1], (1, 0, 0, 0), mode="replicate")
    h = (x + left) * 0.5
    up = F.pad(h[:, :, :-1, :], (0, 0, 1, 0), mode="replicate")
    return (h + up) * 0.5


def map_to_match(score: torch.Tensor) -> dict:
    """Faithful port of the tail of ``continuous_belief.subpixel_match``:
    argmax peak + 3x3 weighted sub-pixel centroid + margin suppression.
    score: (B,28,28) -> dict of (col,row,score,margin,sharpness,heatmap)."""
    from matching import _suppress_margin  # identical margin logic (numpy)

    out = {"col": [], "row": [], "score": [], "margin": [], "sharpness": [],
           "heatmap": []}
    for s in score.detach().float().cpu().numpy():
        h, w = s.shape
        idx = np.unravel_index(int(np.argmax(s)), s.shape)
        r0, c0 = idx
        r_lo, r_hi = max(r0 - 1, 0), min(r0 + 2, h)
        c_lo, c_hi = max(c0 - 1, 0), min(c0 + 2, w)
        win = s[r_lo:r_hi, c_lo:c_hi].astype(np.float64)
        wgt = np.clip(win - win.min(), 0, None) ** 2
        if wgt.sum() <= 0:
            dr = dc = 0.0
        else:
            rr, cc = np.mgrid[r_lo:r_hi, c_lo:c_hi]
            dr = float((wgt * (rr - r0)).sum() / wgt.sum())
            dc = float((wgt * (cc - c0)).sum() / wgt.sum())
        cell = OUT / float(h)
        col = (c0 + 0.5 + dc) * cell
        row = (r0 + 0.5 + dr) * cell
        margin = _suppress_margin(s, idx, 1)
        peak = float(s[r0, c0])
        spread = float(np.clip(peak - win.mean(), 0.0, 1.0))
        out["col"].append(col)
        out["row"].append(row)
        out["score"].append(peak)
        out["margin"].append(margin)
        out["sharpness"].append(spread)
        out["heatmap"].append(s)
    out["col"] = torch.tensor(out["col"], device=score.device)
    out["row"] = torch.tensor(out["row"], device=score.device)
    out["score"] = torch.tensor(out["score"], device=score.device)
    out["margin"] = torch.tensor(out["margin"], device=score.device)
    return out


def pack(out: dict) -> dict:
    """Public interface: (col,row,score,margin) like subpixel_match."""
    return {"col": out["col"], "row": out["row"], "score": out["score"],
            "margin": out["margin"], "map": out["heatmap"],
            "sharpness": out["sharpness"]}


# --------------------------------------------------------------------------
# Fair learned matchers
# --------------------------------------------------------------------------
class PointQueryMatcher(nn.Module):
    """PQ (fair): learnable query on the C1 mechanism.

    q = MLP(desc);  map(c) = cos(q, W_v . pooled_cell(c)).
    With identity weights the map is numerically identical to C1's cosine map.
    """

    def __init__(self, d: int = D, hidden: int = 512):
        super().__init__()
        self.query_mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(inplace=True),
                                       nn.Linear(hidden, d))
        self.value_proj = nn.Linear(d, d)
        self._init_identity()

    def _init_identity(self):
        """Zero the residual path so the untrained model reduces to C1."""
        for m in self.query_mlp:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)

    def match_map(self, obs_feat: torch.Tensor, goal_feat: torch.Tensor):
        desc = goal_desc(goal_feat)                          # (B,512)
        pooled = obs_pooled(obs_feat)                        # (B,512,28,28)
        q = F.normalize(desc + self.query_mlp(desc), dim=1, eps=EPS)
        v = pooled.permute(0, 2, 3, 1) + self.value_proj(
            pooled.permute(0, 2, 3, 1))                      # (B,28,28,512)
        vn = F.normalize(v, dim=-1, eps=EPS)
        sim = torch.einsum("bhwc,bc->bhw", vn, q)            # (B,28,28) cosine
        return sim

    def forward(self, obs_feat: torch.Tensor, goal_feat: torch.Tensor):
        return pack(map_to_match(self.match_map(obs_feat, goal_feat)))


class CrossAttnMatcher(nn.Module):
    """CA (fair): attention-weight match map on the C1 mechanism.

    map(c) = softmax_c( q_proj(desc) . k_proj(pooled_cell(c)) * scale ).
    Everything downstream (peak + centroid + margin + gate) is C1-identical.
    """

    def __init__(self, d: int = D):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.log_scale = nn.Parameter(torch.tensor(0.0))     # scale = e^log_scale

    def match_logits(self, obs_feat: torch.Tensor, goal_feat: torch.Tensor):
        desc = goal_desc(goal_feat)                          # (B,512)
        pooled = obs_pooled(obs_feat).permute(0, 2, 3, 1)    # (B,28,28,512)
        q = self.q_proj(desc)                                # (B,512)
        k = self.k_proj(pooled)                              # (B,28,28,512)
        scale = torch.exp(self.log_scale) / math.sqrt(D)
        return torch.einsum("bhwc,bc->bhw", k, q) * scale    # (B,28,28)

    def match_map(self, obs_feat: torch.Tensor, goal_feat: torch.Tensor):
        logits = self.match_logits(obs_feat, goal_feat)
        return torch.softmax(logits.view(logits.shape[0], -1), dim=1).view_as(logits)

    def forward(self, obs_feat: torch.Tensor, goal_feat: torch.Tensor):
        return pack(map_to_match(self.match_map(obs_feat, goal_feat)))


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def load_matcher(method: str, ckpt: str, device: str = "cuda"):
    m = PointQueryMatcher() if method == "pq" else CrossAttnMatcher()
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except Exception:
        # best.pt may embed a numpy scalar (train_loss); local trusted file.
        state = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(state["model"])
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m
