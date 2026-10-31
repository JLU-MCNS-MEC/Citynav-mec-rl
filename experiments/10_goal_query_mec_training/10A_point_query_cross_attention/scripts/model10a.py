#!/usr/bin/env python3
"""Stage 10A learned matchers: Point Query (PQ) and Cross-Attention (CA).

Both consume the frozen ResNet-50 layer2 feature maps (512 x 28 x 28) of the
north-up observation and of the goal image, and output per-proposal
(col, row, score) in the 224 px north-up frame -- the same interface as
09C's ``subpixel_match`` (top-1 proposal + score margin).

* PQ (learnable point query): a set of learnable query tokens conditioned on
  the goal descriptor cross-attend to the observation feature cells; each
  query predicts an offset proposal and a confidence.
* CA (cross-attention): goal-derived tokens (3x3 block descriptors + global)
  cross-attend to the observation, then refine each other with self-attention;
  each token predicts an offset proposal and a confidence.

Only the heads are trained; the ResNet backbone stays frozen.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

OUT = 224.0
D = 512


def sinusoidal_pos_2d(h: int, w: int, d: int, device) -> torch.Tensor:
    """Fixed sinusoidal 2-D position encoding (DETR-style), (d, h, w)."""
    assert d % 4 == 0
    half = d // 4
    ys = torch.arange(h, device=device, dtype=torch.float32).view(h, 1).repeat(1, w)
    xs = torch.arange(w, device=device, dtype=torch.float32).view(1, w).repeat(h, 1)
    emb = []
    for v, scale in ((ys, h), (xs, w)):
        freqs = torch.arange(half, device=device, dtype=torch.float32)
        freqs = 1.0 / (scale ** (freqs / max(half - 1, 1)))
        ang = v.unsqueeze(-1) * freqs
        emb.append(torch.sin(ang))
        emb.append(torch.cos(ang))
    return torch.cat(emb, dim=-1).permute(2, 0, 1)  # (d, h, w)


class _ProposalHead(nn.Module):
    """token -> (col, row, score_logit); col/row in [0,223], logit unconstrained."""

    def __init__(self, d: int = D):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(inplace=True),
                                 nn.Linear(256, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        out = torch.cat([out[..., :2].sigmoid() * (OUT - 1), out[..., 2:]], dim=-1)
        return out


def pick_top(out: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """(B,N,3) [col,row,logit] -> top-1 (col,row,score) and margin = s1 - s2."""
    scores = out[..., 2].sigmoid()
    s1, i1 = scores.max(dim=1)
    masked = scores.scatter(1, i1.unsqueeze(1),
                            torch.full_like(scores[:, :1], -1e9))
    s2 = masked.max(dim=1).values
    top = out.gather(1, i1.unsqueeze(1).unsqueeze(-1).expand(-1, 1, 3)).squeeze(1)
    return top[..., 0], top[..., 1], s1, (s1 - s2)


class PointQueryMatcher(nn.Module):
    """Learnable point queries conditioned on the goal descriptor."""

    def __init__(self, d: int = D, n_queries: int = 12, heads: int = 8):
        super().__init__()
        self.n_queries = n_queries
        self.query = nn.Parameter(torch.randn(n_queries, d) * 0.02)
        self.anchor = nn.Parameter(torch.zeros(n_queries, 2))
        self.anchor_proj = nn.Sequential(nn.Linear(2, d), nn.GELU())
        self.goal_proj = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.head = _ProposalHead(d)

    def forward(self, obs_feat: torch.Tensor, goal_feat: torch.Tensor):
        """obs/goal: (B,512,28,28) -> dict with top-1 (col,row,score,margin)."""
        b = obs_feat.shape[0]
        pos = sinusoidal_pos_2d(obs_feat.shape[2], obs_feat.shape[3], D,
                                obs_feat.device).flatten(1).T  # (HW,d)
        obs_tokens = obs_feat.flatten(2).permute(0, 2, 1)       # (B,HW,d)
        kv = obs_tokens + pos.unsqueeze(0)
        g = goal_feat.mean(dim=(2, 3))                          # (B,d)
        ctx = self.goal_proj(g).unsqueeze(1)                    # (B,1,d)
        anc = torch.tanh(self.anchor)
        q = self.query.unsqueeze(0) + ctx + self.anchor_proj(anc).unsqueeze(0)
        q2, attn = self.cross(q, kv, kv)
        q2 = self.norm(q + q2)
        out = self.head(q2)                                     # (B,N,3)
        col, row, score, margin = pick_top(out)
        return {"col": col, "row": row, "score": score, "margin": margin,
                "out": out, "attn": attn, "queries": q2, "anchor": anc}


class CrossAttnMatcher(nn.Module):
    """Goal-derived tokens (3x3 block descriptors + global) with cross-attention."""

    def __init__(self, d: int = D, heads: int = 8):
        super().__init__()
        self.n_tokens = 10
        self.pos_pe = nn.Parameter(torch.randn(self.n_tokens, d) * 0.02)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.head = _ProposalHead(d)

    def goal_tokens(self, goal_feat: torch.Tensor) -> torch.Tensor:
        """(B,512,28,28) -> (B,10,512): 3x3 block means + global mean."""
        b = goal_feat.shape[0]
        h = w = goal_feat.shape[2]
        tokens = []
        for r in range(3):
            for c in range(3):
                r0, r1 = r * h // 3, (r + 1) * h // 3
                c0, c1 = c * w // 3, (c + 1) * w // 3
                tokens.append(goal_feat[:, :, r0:r1, c0:c1].mean(dim=(2, 3)))
        tokens.append(goal_feat.mean(dim=(2, 3)))
        return torch.stack(tokens, dim=1)                       # (B,10,d)

    def forward(self, obs_feat: torch.Tensor, goal_feat: torch.Tensor):
        b = obs_feat.shape[0]
        pos = sinusoidal_pos_2d(obs_feat.shape[2], obs_feat.shape[3], D,
                                obs_feat.device).flatten(1).T
        obs_tokens = obs_feat.flatten(2).permute(0, 2, 1)
        kv = obs_tokens + pos.unsqueeze(0)
        q = self.goal_tokens(goal_feat) + self.pos_pe.unsqueeze(0)
        q2, attn_cross = self.cross(q, kv, kv)
        q2 = self.norm1(q + q2)
        q3, attn_self = self.self_attn(q2, q2, q2)
        q3 = self.norm2(q2 + q3)
        out = self.head(q3)
        col, row, score, margin = pick_top(out)
        return {"col": col, "row": row, "score": score, "margin": margin,
                "out": out, "attn_cross": attn_cross, "attn_self": attn_self,
                "tokens": q3}


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
