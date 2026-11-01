#!/usr/bin/env python3
"""Transformer-M v5 (11C): M-v4 + multi-landmark geometry tokens.

Everything identical to M-v4 (model11b.py) except two extra token streams
built from the per-step top-K 09C belief mixture state (saved by
build_features_11b.py as ``landmarks`` (T,8,3) = [mean_x, mean_y, weight]):

* landmark tokens (8): per-landmark [distance, bearing, weight] relative to
  the current pose in the heading frame, projected to d_model;
* relation token (1): pairwise spatial relations among active landmarks
  (weight > 0): pairwise distance and angle difference for all 28 pairs,
  zero for inactive pairs.

These add ONLY geometry available to C1/09C at inference time (doc 2.2) --
no true target coordinates, no success flags.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model11b import (BELIEF_IDX, MATCH_IDX, N_ACTIONS, N_HISTORY, N_STATE,
                      N_TARGET, N_VIS_TOKENS, SPATIAL_CH, TransformerM11)

N_LM = 8
N_REL = N_LM * (N_LM - 1) // 2 * 2          # 28 pairs x (dist, angle-diff)
N_TYPES = 7                                  # +5 landmark, +6 relation


class TransformerM12(nn.Module):
    """M-v5: TransformerM11 + multi-landmark geometry tokens."""

    def __init__(self, d_model: int = 256, n_layers: int = 4, n_heads: int = 8,
                 ffn: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.vis_proj = nn.Linear(SPATIAL_CH, d_model)
        self.vis_pos = nn.Parameter(torch.zeros(N_VIS_TOKENS, d_model))
        self.target_proj = nn.Linear(N_TARGET, d_model)
        self.hist_vis = nn.Linear(SPATIAL_CH, d_model)
        self.hist_pose = nn.Sequential(nn.Linear(3, d_model), nn.GELU(),
                                       nn.Linear(d_model, d_model))
        self.hist_action = nn.Embedding(N_ACTIONS, d_model)
        self.hist_time = nn.Parameter(torch.zeros(N_HISTORY, d_model))
        self.state_proj = nn.Sequential(nn.Linear(N_STATE, d_model), nn.GELU(),
                                        nn.Linear(d_model, d_model))
        self.lm_proj = nn.Sequential(nn.Linear(3, d_model), nn.GELU(),
                                     nn.Linear(d_model, d_model))
        self.rel_proj = nn.Sequential(nn.Linear(N_REL, d_model), nn.GELU(),
                                      nn.Linear(d_model, d_model))
        self.type_emb = nn.Embedding(N_TYPES, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N_ACTIONS)

    @staticmethod
    def landmark_geometry(landmarks: torch.Tensor, states: torch.Tensor,
                          yaw_only: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-landmark [dist, bearing, weight] + pairwise relations.

        landmarks: (B,8,3) [mx, my, w]; states: (B,3) [x, y, yaw].
        Returns (lm_feats (B,8,3), rel_feats (B,56)).
        """
        b = landmarks.shape[0]
        mx, my, w = landmarks[..., 0], landmarks[..., 1], landmarks[..., 2]
        dx = mx - states[:, 0:1]
        dy = my - states[:, 1:2]
        yaw = states[:, 2:3]
        fwd = dx * torch.cos(yaw) + dy * torch.sin(yaw)
        lat = -dx * torch.sin(yaw) + dy * torch.cos(yaw)
        dist = torch.sqrt(fwd * fwd + lat * lat)
        bearing = torch.atan2(lat, fwd)
        active = (w > 0).float()
        lm_feats = torch.stack([dist, bearing, w], dim=-1)
        rel = torch.zeros(b, N_REL, device=landmarks.device)
        k = 0
        for i in range(N_LM):
            for j in range(i + 1, N_LM):
                d = torch.sqrt((mx[:, i] - mx[:, j]) ** 2 + (my[:, i] - my[:, j]) ** 2)
                ang = torch.atan2(my[:, j] - my[:, i], mx[:, j] - mx[:, i])
                mask = active[:, i] * active[:, j]
                rel[:, k] = d * mask
                rel[:, k + 1] = ang * mask
                k += 2
        return lm_feats, rel

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        b = batch["current"].shape[0]
        device = batch["current"].device
        tokens, masks = [], []

        tokens.append((self.cls + self.type_emb.weight[0]).expand(b, 1, -1))
        masks.append(torch.zeros(b, 1, dtype=torch.bool, device=device))

        vis = (self.vis_proj(batch["current"].reshape(b, N_VIS_TOKENS, SPATIAL_CH))
               + self.vis_pos[None] + self.type_emb.weight[1])
        tokens.append(vis)
        masks.append(torch.zeros(b, N_VIS_TOKENS, dtype=torch.bool, device=device))

        target = (self.target_proj(batch["target_emb"]) + self.type_emb.weight[2])[:, None, :]
        tokens.append(target)
        masks.append(torch.zeros(b, 1, dtype=torch.bool, device=device))

        hist = (self.hist_vis(batch["history_vis"])
                + self.hist_pose(batch["history_pose"])
                + self.hist_action(batch["history_action"].argmax(-1))
                + self.hist_time[None]
                + self.type_emb.weight[3])
        tokens.append(hist)
        masks.append(batch["history_mask"] < 0.5)

        belief = batch["belief"][:, BELIEF_IDX]
        match = batch["match"][:, MATCH_IDX]
        state = torch.cat([belief, match], dim=-1)
        state = (self.state_proj(state) + self.type_emb.weight[4])[:, None, :]
        tokens.append(state)
        masks.append(torch.zeros(b, 1, dtype=torch.bool, device=device))

        lm_feats, rel = self.landmark_geometry(batch["landmarks"], batch["states"])
        lm_tokens = self.lm_proj(lm_feats) + self.type_emb.weight[5]
        tokens.append(lm_tokens)
        masks.append(torch.zeros(b, N_LM, dtype=torch.bool, device=device))
        rel_token = (self.rel_proj(rel) + self.type_emb.weight[6])[:, None, :]
        tokens.append(rel_token)
        masks.append(torch.zeros(b, 1, dtype=torch.bool, device=device))

        seq = torch.cat(tokens, dim=1)
        pad = torch.cat(masks, dim=1)
        out = self.encoder(seq, src_key_padding_mask=pad)
        return self.head(self.norm(out[:, 0]))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
