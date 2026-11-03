#!/usr/bin/env python3
"""Medium MEC（四动作版）：GoalCrossAttentionMEC 架构适配 V5 完整多模态输入。

核心 = goal→current 交叉注意力（V8.2 TransformerMMEC 的 goal_current_attn）：
目标描述子作为 query，对 49 个 current 视觉 token 做多头注意力（无自注意力
encoder，保持非 Transformer 融合）；随后与历史/状态/几何 late fusion 成四动作
logits。历史用 masked mean-pool；landmarks/relation 用修正后
TransformerM12Fix.landmark_geometry（与 V5fix 同款几何）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model11b import (BELIEF_IDX, MATCH_IDX, N_TARGET, SPATIAL_CH)
from model11c_fix import TransformerM12Fix


class MediumMEC4(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.cur_proj = nn.Linear(SPATIAL_CH, d_model)
        self.goal_proj = nn.Linear(N_TARGET, d_model)
        self.goal_current_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.0, batch_first=True)
        self.gc_norm = nn.LayerNorm(d_model)
        self.gc_ffn = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                    nn.Linear(2 * d_model, d_model))
        self.gc_ffn_norm = nn.LayerNorm(d_model)
        self.hist_vis_proj = nn.Sequential(nn.Linear(SPATIAL_CH, d_model),
                                           nn.GELU())
        self.hist_pose_proj = nn.Sequential(nn.Linear(3, d_model), nn.GELU())
        self.hist_act_proj = nn.Sequential(nn.Linear(4, d_model), nn.GELU())
        self.state_proj = nn.Sequential(nn.Linear(12, d_model), nn.GELU())
        self.lm_proj = nn.Sequential(nn.Linear(3, d_model), nn.GELU())
        self.rel_proj = nn.Sequential(nn.Linear(3, d_model), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(7 * d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, 4))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        b = batch["current"].shape[0]
        cur = self.cur_proj(batch["current"].reshape(b, -1, SPATIAL_CH))
        goal = self.goal_proj(batch["target_emb"])[:, None, :]
        matched, _ = self.goal_current_attn(query=goal, key=cur, value=cur)
        matched = self.gc_norm(goal + matched)
        matched = self.gc_ffn_norm(matched + self.gc_ffn(matched))
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
        parts = [matched.squeeze(1),
                 self.hist_vis_proj(hvis),
                 self.hist_pose_proj(hpose),
                 self.hist_act_proj(hact),
                 self.state_proj(state),
                 self.lm_proj(lm_feats.mean(dim=1)),
                 self.rel_proj(rel.mean(dim=1))]
        return self.head(torch.cat(parts, dim=-1))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
