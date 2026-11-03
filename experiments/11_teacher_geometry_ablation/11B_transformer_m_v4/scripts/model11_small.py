#!/usr/bin/env python3
"""Small MEC（四动作版）：V8.2 SmallMEC（MeanPoolMLP）架构适配 V5 完整多模态输入。

非 Transformer：late-fusion MLP。各模态：
- current 视觉 token 均值池化（MeanPool 本体）；
- target 描述子、belief/match 去重 12 维状态；
- 历史（vis/pose/action）按 history_mask 做 masked mean-pool（padding 不参与）；
- landmarks / relation 使用修正后的 TransformerM12Fix.landmark_geometry
  （修正 1-4 同款几何：mask、相对角、归一化、槽位重排），无效槽已置零。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model11b import (BELIEF_IDX, MATCH_IDX, N_TARGET, SPATIAL_CH)
from model11c_fix import TransformerM12Fix


class SmallMEC4(nn.Module):
    def __init__(self, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.cur_proj = nn.Sequential(nn.Linear(SPATIAL_CH, d_model), nn.GELU())
        self.target_proj = nn.Sequential(nn.Linear(N_TARGET, d_model), nn.GELU())
        self.hist_vis_proj = nn.Sequential(nn.Linear(SPATIAL_CH, d_model),
                                           nn.GELU())
        self.hist_pose_proj = nn.Sequential(nn.Linear(3, d_model), nn.GELU())
        self.hist_act_proj = nn.Sequential(nn.Linear(4, d_model), nn.GELU())
        self.state_proj = nn.Sequential(nn.Linear(12, d_model), nn.GELU())
        self.lm_proj = nn.Sequential(nn.Linear(3, d_model), nn.GELU())
        self.rel_proj = nn.Sequential(nn.Linear(3, d_model), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(8 * d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, 4))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        b = batch["current"].shape[0]
        cur = batch["current"].reshape(b, -1, SPATIAL_CH).mean(dim=1)
        hmask = batch["history_mask"]
        denom = hmask.sum(dim=1, keepdim=True).clamp(min=1.0)
        hvis = (batch["history_vis"] * hmask[..., None]).sum(dim=1) / denom
        hpose = (batch["history_pose"] * hmask[..., None]).sum(dim=1) / denom
        hact = (batch["history_action"] * hmask[..., None]).sum(dim=1) / denom
        belief = batch["belief"][:, BELIEF_IDX]
        match = batch["match"][:, MATCH_IDX]
        state = torch.cat([belief, match], dim=-1)
        lm_feats, rel, _, _ = TransformerM12Fix.landmark_geometry(
            batch["landmarks"], batch["states"])
        parts = [self.cur_proj(cur),
                 self.target_proj(batch["target_emb"]),
                 self.hist_vis_proj(hvis),
                 self.hist_pose_proj(hpose),
                 self.hist_act_proj(hact),
                 self.state_proj(state),
                 self.lm_proj(lm_feats.mean(dim=1)),
                 self.rel_proj(rel.mean(dim=1))]
        return self.head(torch.cat(parts, dim=-1))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
