#!/usr/bin/env python3
"""Transformer-M for fixed-50 m four-action visual goal navigation.

All inputs are fused into one token sequence and every action (including STOP)
is predicted from the fused CLS representation.  There is no separate STOP
head and no language/BERT/network-state input.
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_ACTIONS = 4
SPATIAL_CH = 2048
SPATIAL_HW = 7
N_VIS_TOKENS = SPATIAL_HW * SPATIAL_HW
N_HISTORY = 16
N_CANDIDATES = 5


class TransformerM(nn.Module):
    def __init__(self, d_model: int = 256, n_layers: int = 4, n_heads: int = 8,
                 ffn: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.vis_proj = nn.Linear(SPATIAL_CH, d_model)
        self.vis_pos = nn.Parameter(torch.zeros(N_VIS_TOKENS, d_model))
        self.target_proj = nn.Linear(SPATIAL_CH, d_model)
        self.hist_vis = nn.Linear(SPATIAL_CH, d_model)
        self.hist_pose = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.hist_action = nn.Embedding(N_ACTIONS, d_model)
        self.cand_vis = nn.Linear(SPATIAL_CH, d_model)
        self.cand_meta = nn.Sequential(nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.belief_proj = nn.Sequential(nn.Linear(12, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.type_emb = nn.Embedding(6, d_model)  # 0 CLS, 1 visual, 2 target, 3 history, 4 candidate, 5 belief
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
        tokens = []
        masks = []

        cls = (self.cls + self.type_emb.weight[0]).expand(b, 1, -1)
        tokens.append(cls)
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
                + self.hist_action(batch["history_action"])
                + self.type_emb.weight[3])
        tokens.append(hist)
        masks.append(batch["history_mask"] < 0.5)

        cand = (self.cand_vis(batch["candidate_emb"])
                + self.cand_meta(batch["candidate_meta"])
                + self.type_emb.weight[4])
        tokens.append(cand)
        masks.append(torch.zeros(b, N_CANDIDATES, dtype=torch.bool, device=device))

        belief = (self.belief_proj(batch["belief"]) + self.type_emb.weight[5])[:, None, :]
        tokens.append(belief)
        masks.append(torch.zeros(b, 1, dtype=torch.bool, device=device))

        seq = torch.cat(tokens, dim=1)
        pad = torch.cat(masks, dim=1)
        out = self.encoder(seq, src_key_padding_mask=pad)
        return self.head(self.norm(out[:, 0]))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
