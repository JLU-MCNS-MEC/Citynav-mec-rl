#!/usr/bin/env python3
"""Transformer-S / Large（四动作版）：MEC-v8.2 ExplicitReasoningTransformer 适配
V5 完整多模态输入（简报第 2 阶段）。

原始定义：mec-v8.2/gbs_model_v3.py `explicit_transformer` 架构（原样保留）：
- 视觉投影 + type/pos 编码 + cls_token
- goal→current 交叉注意力块（LayerNorm + ff_mult FFN）
- fusion TransformerEncoder（d_model=hidden, nhead, ff=hidden*ff_mult,
  gelu, norm_first, dropout 0.1）
- final_norm + 输出头

尺寸（mec-v8.2 config.py GBS_MODEL_CONFIGS，原样）：
- Transformer-S: hidden_dim=192, num_layers=1, num_heads=4, ff_mult=3
- Large:        hidden_dim=352, num_layers=2, num_heads=8, ff_mult=4

适配（只动输入投影与融合维度，不动架构骨架）：
- 四动作头（Stop/Forward/TurnLeft/TurnRight，fixed50m 协议）
- 输入序列 = cls + 49 current 视觉 + 1 goal + 1 历史（16 步 masked mean-pool
  的 vis+pose+act 三投影求和）+ 1 状态（belief+match）+ 8 几何 + 28 关系
  = 89 token；几何/关系 mask（True=pad）并入 src_key_padding_mask
- 几何/关系特征用 TransformerM12Fix.landmark_geometry（与 V5fix 同款修正）
- 移除 v8.2 的 heatmap/network 模态（fixed50m 协议无此输入）
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model11b import BELIEF_IDX, MATCH_IDX, N_TARGET, SPATIAL_CH
from model11c_fix import TransformerM12Fix

N_CURRENT_TOKENS = 49
N_LM_TOKENS = 8
N_REL_TOKENS = 28


class ExplicitReasoningMEC4(nn.Module):
    """v8.2 explicit_transformer 骨架的四动作适配（Transformer-S/Large 基类）。"""

    hidden_dim: int = 192
    num_layers: int = 1
    num_heads: int = 4
    ff_mult: int = 3

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        h, nh, ff = self.hidden_dim, self.num_heads, self.ff_mult
        if h % nh != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.visual_proj = nn.Linear(SPATIAL_CH, h)
        self.goal_proj = nn.Linear(N_TARGET, h)
        self.current_type = nn.Parameter(torch.zeros(1, 1, h))
        self.goal_type = nn.Parameter(torch.zeros(1, 1, h))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, h))
        self.current_pos = nn.Parameter(torch.zeros(1, N_CURRENT_TOKENS, h))
        self.goal_pos = nn.Parameter(torch.zeros(1, 1, h))

        self.goal_current_attn = nn.MultiheadAttention(
            h, nh, dropout=dropout, batch_first=True)
        self.gc_norm = nn.LayerNorm(h)
        self.gc_ffn = nn.Sequential(
            nn.Linear(h, h * ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h * ff, h))
        self.gc_ffn_norm = nn.LayerNorm(h)

        self.hist_vis_proj = nn.Sequential(nn.Linear(SPATIAL_CH, h), nn.GELU())
        self.hist_pose_proj = nn.Sequential(nn.Linear(3, h), nn.GELU())
        self.hist_act_proj = nn.Sequential(nn.Linear(4, h), nn.GELU())
        self.state_proj = nn.Sequential(nn.Linear(12, h), nn.GELU())
        self.lm_proj = nn.Sequential(nn.Linear(3, h), nn.GELU())
        self.rel_proj = nn.Sequential(nn.Linear(3, h), nn.GELU())

        self.fusion_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=h, nhead=nh, dim_feedforward=h * ff,
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True),
            num_layers=self.num_layers)
        self.final_norm = nn.LayerNorm(h)
        self.head = nn.Linear(h, 4)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        b = batch["current"].shape[0]
        cur = (self.visual_proj(batch["current"].reshape(b, -1, SPATIAL_CH))
               + self.current_pos + self.current_type)
        goal = (self.goal_proj(batch["target_emb"])[:, None, :]
                + self.goal_pos + self.goal_type)
        matched, _ = self.goal_current_attn(query=goal, key=cur, value=cur)
        matched = self.gc_norm(goal + matched)
        matched = self.gc_ffn_norm(matched + self.gc_ffn(matched))

        hmask = batch["history_mask"]
        denom = hmask.sum(dim=1, keepdim=True).clamp(min=1.0)
        hvis = (batch["history_vis"] * hmask[..., None]).sum(dim=1) / denom
        hpose = (batch["history_pose"] * hmask[..., None]).sum(dim=1) / denom
        hact = (batch["history_action"] * hmask[..., None]).sum(dim=1) / denom
        hist = (self.hist_vis_proj(hvis) + self.hist_pose_proj(hpose)
                + self.hist_act_proj(hact))[:, None, :]

        state = torch.cat([batch["belief"][:, BELIEF_IDX],
                           batch["match"][:, MATCH_IDX]], dim=-1)
        state = self.state_proj(state)[:, None, :]

        lm_feats, rel_feats, lm_mask, rel_mask = \
            TransformerM12Fix.landmark_geometry(batch["landmarks"],
                                                batch["states"])
        lm_tok = self.lm_proj(lm_feats)
        rel_tok = self.rel_proj(rel_feats)

        cls_tok = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls_tok, cur, matched, hist, state, lm_tok, rel_tok],
                        dim=1)
        # 前缀 token 数 = cls 1 + current 49 + goal 1 + hist 1 + state 1 = 53
        pad = torch.cat([torch.zeros(b, 53, dtype=torch.bool,
                                     device=lm_mask.device),
                         lm_mask, rel_mask], dim=1)
        assert pad.shape[1] == seq.shape[1], "mask/seq token 数不一致"
        out = self.fusion_encoder(seq, src_key_padding_mask=pad)
        return self.head(self.final_norm(out[:, 0]))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TransformerS4(ExplicitReasoningMEC4):
    """Transformer-S 四动作版（192 / 1 层 / 4 头 / ff×3，v8.2 原尺寸）。"""

    hidden_dim = 192
    num_layers = 1
    num_heads = 4
    ff_mult = 3


class TransformerL4(ExplicitReasoningMEC4):
    """Large 四动作版（352 / 2 层 / 8 头 / ff×4，v8.2 原尺寸）。"""

    hidden_dim = 352
    num_layers = 2
    num_heads = 8
    ff_mult = 4
