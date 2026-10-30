#!/usr/bin/env python3
"""Transformer-M v2 for fixed-50 m four-action visual goal navigation.

Input contract (stage 05):

| key            | shape        | meaning                                    |
|----------------|--------------|--------------------------------------------|
| current        | B,7,7,2048   | heading-forward local RGB, frozen ResNet   |
| target_emb     | B,2048       | target reference embedding (image only)    |
| history_vis    | B,16,2048    | pooled visual of states t-16..t-1          |
| history_pose   | B,16,3       | relative pose in the current heading frame |
| history_action | B,16,4       | one-hot of the past four actions           |
| history_mask   | B,16         | 1 for a real past step, 0 for padding      |
| belief         | B,12         | global goal belief (visual match + geometry)|

All tokens are fused in one sequence; STOP is one of the four output classes of
the same head (no separate STOP head).  No language, BERT, network state, true
target coordinates, target id or target distance enters the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_ACTIONS = 4
SPATIAL_CH = 2048
SPATIAL_HW = 7
N_VIS_TOKENS = SPATIAL_HW * SPATIAL_HW
N_HISTORY = 16
N_BELIEF = 12
# token types: 0 CLS, 1 visual, 2 target, 3 history, 4 belief
N_TYPES = 5


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
        self.belief_proj = nn.Sequential(nn.Linear(N_BELIEF, d_model), nn.GELU(),
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
                + self.type_emb.weight[3])
        tokens.append(hist)
        masks.append(batch["history_mask"] < 0.5)

        belief = (self.belief_proj(batch["belief"]) + self.type_emb.weight[4])[:, None, :]
        tokens.append(belief)
        masks.append(torch.zeros(b, 1, dtype=torch.bool, device=device))

        seq = torch.cat(tokens, dim=1)
        pad = torch.cat(masks, dim=1)
        out = self.encoder(seq, src_key_padding_mask=pad)
        return self.head(self.norm(out[:, 0]))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
