#!/usr/bin/env python3
"""Transformer-M v4 (11B) for fixed-50 m four-action navigation.

Same backbone as M-v3 (model10c.py) with the three approved v4 fixes:

1. history time position encoding: learned per-slot time embedding (16 slots,
   slot k = k steps ago) added to the history tokens; v3 had only a shared
   type embedding, so the transformer could not tell recency apart;
2. deduplicated state token: v3 fused belief(10) + match(5) = 15-d, but
   dx/dy/score appeared twice (belief[8:10]==match[0:2], belief[4]==match[2]);
   v4 state = 12-d: [fwd, lat, dist, bearing, unc, ent, since] (belief
   position/uncertainty) + [dx, dy, score, margin, sharpness] (raw C1
   measurement), each quantity exactly once;
3. training schedule fix lives in train_11b.py (per-batch OneCycleLR instead
   of per-epoch stepping).

Input contract identical to M-v3 otherwise (current 49x2048 layer4 spatial,
C1 goal descriptor 512, causal 16-step history, 09C belief + C1 match fused
into ONE deduplicated state token).  No multi-landmark geometry (that is v5).
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_ACTIONS = 4
SPATIAL_CH = 2048
SPATIAL_HW = 7
N_VIS_TOKENS = SPATIAL_HW * SPATIAL_HW
N_HISTORY = 16
N_BELIEF = 10
N_MATCH = 5
# v4 deduplicated state: belief position/uncertainty (fwd,lat,dist,bearing,
# unc,ent,since) + raw match (dx,dy,score,margin,sharpness)
N_STATE = 12
N_TARGET = 512
# token types: 0 CLS, 1 visual, 2 target, 3 history, 4 state
N_TYPES = 5

# index maps to slice the deduplicated state out of the stored arrays
BELIEF_IDX = [0, 1, 2, 3, 5, 6, 7]          # fwd,lat,dist,bearing,unc,ent,since
MATCH_IDX = [0, 1, 2, 3, 4]                 # dx,dy,score,margin,sharpness


class TransformerM11(nn.Module):
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
        self.hist_time = nn.Parameter(torch.zeros(N_HISTORY, d_model))  # v4 fix 1
        self.state_proj = nn.Sequential(nn.Linear(N_STATE, d_model), nn.GELU(),
                                        nn.Linear(d_model, d_model))   # v4 fix 2
        self.type_emb = nn.Embedding(N_TYPES, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N_ACTIONS)

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
                + self.hist_time[None]          # v4 fix 1: per-slot time embedding
                + self.type_emb.weight[3])
        tokens.append(hist)
        masks.append(batch["history_mask"] < 0.5)

        belief = batch["belief"][:, BELIEF_IDX]
        match = batch["match"][:, MATCH_IDX]
        state = torch.cat([belief, match], dim=-1)          # v4 fix 2: 12-d
        state = (self.state_proj(state) + self.type_emb.weight[4])[:, None, :]
        tokens.append(state)
        masks.append(torch.zeros(b, 1, dtype=torch.bool, device=device))

        seq = torch.cat(tokens, dim=1)
        pad = torch.cat(masks, dim=1)
        out = self.encoder(seq, src_key_padding_mask=pad)
        return self.head(self.norm(out[:, 0]))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
