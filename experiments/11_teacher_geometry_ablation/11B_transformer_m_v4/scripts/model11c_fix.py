#!/usr/bin/env python3
"""Transformer-M v5 修正版（11C-fix）：TransformerM12 六项修正的模型侧实现（修正 1-4）。

旧 model11c.py 字节级冻结：旧权重 runs_11c/seed42/best.pt 的 state_dict 只能由旧类
加载（rel_proj 输入维度 56→3 的变化会使旧权重永久不可加载），旧报告脚本
（run_11d、replay_corrected）继续 import 旧类不受影响。

与旧实现的差异：
1. 无效几何 token 的 attention mask：lm_mask/rel_mask（True=pad，与 history 同约定）
   并入 src_key_padding_mask；无效槽特征显式置 0（旧实现把空槽算成指向世界原点的
   垃圾 dist/bearing）；
2. 关系角 = heading 帧 bearing 差 wrap 到 (-π,π]（旧实现为世界帧绝对方位角，
   泄漏全局朝向、依赖槽序）；dist÷DIST_SCALE_M、角÷ANGLE_SCALE 归一化；
3. relation 由单 56 维 token 改为 28 个 per-pair token（特征 [d/50, Δ/π, valid]），
   无效对特征全 0 且被 mask；合法合并对（d=0）特征 [0,0,1] 与无效对结构可区分；
4. 槽位按权重降序稳定重排（并列保持 belief 原序，空槽 w=0 排尾）——存储
   landmarks [mx,my,w] 语义不变，无需重建特征（CACHE_VERSION 不 bump）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from model11b import (BELIEF_IDX, MATCH_IDX, N_ACTIONS, N_HISTORY, N_STATE,
                      N_TARGET, N_VIS_TOKENS, SPATIAL_CH)
from v5fix_contract import ANGLE_SCALE, DIST_SCALE_M

N_LM = 8
N_PAIRS = N_LM * (N_LM - 1) // 2        # 28 对
N_REL_FEAT = 3                           # [d/50, Δ/π, valid]
N_TYPES = 7                              # 0 cls, 1 vis, 2 target, 3 hist, 4 state, 5 lm, 6 rel


def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
    """wrap 到 (-π, π]（π → -π）。"""
    return (x + math.pi) % (2 * math.pi) - math.pi


class TransformerM12Fix(nn.Module):
    """M-v5 修正版：TransformerM12 几何层修正 1-4。"""

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
        self.rel_proj = nn.Sequential(nn.Linear(N_REL_FEAT, d_model), nn.GELU(),
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
    def landmark_geometry(landmarks: torch.Tensor, states: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor,
                                     torch.Tensor, torch.Tensor]:
        """修正 1-4 的几何层。

        landmarks: (B,8,3) [mx, my, w]; states: (B,3) [x, y, yaw]。
        返回 (lm_feats (B,8,3), rel_feats (B,28,3), lm_mask (B,8), rel_mask (B,28))：
        lm_feats 每槽 [dist/50, bearing/π, w]（heading 帧、左正）；
        rel_feats 每对 [d/50, Δbearing/π, valid]；
        mask 约定 True=pad（与 history_mask<0.5 及 src_key_padding_mask 语义一致）。
        """
        b = landmarks.shape[0]
        # 修正 4：按权重降序稳定重排（并列保持 belief 原序；空槽 w=0 自然排尾）
        order = torch.argsort(-landmarks[..., 2], dim=1, stable=True)
        lm = torch.gather(landmarks, 1,
                          order.unsqueeze(-1).expand(-1, -1, landmarks.shape[-1]))
        mx, my, w = lm[..., 0], lm[..., 1], lm[..., 2]
        dx = mx - states[:, 0:1]
        dy = my - states[:, 1:2]
        yaw = states[:, 2:3]
        fwd = dx * torch.cos(yaw) + dy * torch.sin(yaw)
        lat = -dx * torch.sin(yaw) + dy * torch.cos(yaw)
        dist = torch.sqrt(fwd * fwd + lat * lat)
        bearing = torch.atan2(lat, fwd)
        active = w > 0
        # 修正 1：无效槽特征显式置 0（旧实现为指向世界原点的垃圾值）
        dist = torch.where(active, dist, torch.zeros_like(dist))
        bearing = torch.where(active, bearing, torch.zeros_like(bearing))
        lm_feats = torch.stack([dist / DIST_SCALE_M, bearing / ANGLE_SCALE, w],
                               dim=-1)
        lm_mask = ~active
        # 修正 2+3：per-pair token（重排保证 i<j 时 w_i >= w_j）
        rel = torch.zeros(b, N_PAIRS, N_REL_FEAT, device=landmarks.device)
        k = 0
        for i in range(N_LM):
            for j in range(i + 1, N_LM):
                d = torch.sqrt((mx[:, i] - mx[:, j]) ** 2
                               + (my[:, i] - my[:, j]) ** 2)
                dang = _wrap_pi(bearing[:, j] - bearing[:, i])
                valid = active[:, i] & active[:, j]
                rel[:, k, 0] = torch.where(valid, d / DIST_SCALE_M,
                                           torch.zeros_like(d))
                rel[:, k, 1] = torch.where(valid, dang / ANGLE_SCALE,
                                           torch.zeros_like(dang))
                rel[:, k, 2] = valid.float()
                k += 1
        rel_mask = ~(rel[..., 2] > 0.5)
        return lm_feats, rel, lm_mask, rel_mask

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

        target = (self.target_proj(batch["target_emb"])
                  + self.type_emb.weight[2])[:, None, :]
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

        lm_feats, rel, lm_mask, rel_mask = self.landmark_geometry(
            batch["landmarks"], batch["states"])
        lm_tokens = self.lm_proj(lm_feats) + self.type_emb.weight[5]
        tokens.append(lm_tokens)
        masks.append(lm_mask)
        rel_tokens = self.rel_proj(rel) + self.type_emb.weight[6]
        tokens.append(rel_tokens)
        masks.append(rel_mask)

        seq = torch.cat(tokens, dim=1)
        pad = torch.cat(masks, dim=1)
        out = self.encoder(seq, src_key_padding_mask=pad)
        return self.head(self.norm(out[:, 0]))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
