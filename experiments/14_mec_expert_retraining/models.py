#!/usr/bin/env python3
"""Stage 14: the five R2 preset MEC architectures (fresh training).

These classes are the R2 presets, built for retraining from scratch:
  Small      : MLP + per-token visual MLP 2048->12288->2048, D=512, head 2048
  Medium     : goal cross-attention + per-token MLP 2048->36864->2048,
               D=512, gc FFN 4096, head 2048
  Transformer-S : 104-token explicit Transformer, D=1408, L=8, H=8,
                  gc FFN 4224, fusion FFN 5376
  Transformer-M : TransformerM12Fix, D=2048, L=10, H=16, FFN 7424
  Large         : 104-token explicit Transformer, D=2560, L=12, H=16,
                  gc FFN 10240, fusion FFN 10240

No original model definition or checkpoint is modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments/11_teacher_geometry_ablation"
                       "/11B_transformer_m_v4/scripts"))

from model11b import BELIEF_IDX, MATCH_IDX  # noqa: E402
from model11c_fix import TransformerM12Fix  # noqa: E402

DROPOUT = 0.1


class SmallPreset(nn.Module):
    def __init__(self, d=512, tok=12288, head_h=2048):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(2048, tok), nn.GELU(), nn.Linear(tok, 2048))
        self.cur_proj = nn.Sequential(nn.Linear(2048, d), nn.GELU())
        self.target_proj = nn.Sequential(nn.Linear(512, d), nn.GELU())
        self.hist_vis_proj = nn.Sequential(nn.Linear(2048, d), nn.GELU())
        self.hist_pose_proj = nn.Sequential(nn.Linear(3, d), nn.GELU())
        self.hist_act_proj = nn.Sequential(nn.Linear(4, d), nn.GELU())
        self.state_proj = nn.Sequential(nn.Linear(12, d), nn.GELU())
        self.lm_proj = nn.Sequential(nn.Linear(3, d), nn.GELU())
        self.rel_proj = nn.Sequential(nn.Linear(3, d), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(8 * d, head_h), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(head_h, 4))

    def forward(self, batch):
        b = batch["current"].shape[0]
        cur = self.token_mlp(batch["current"].reshape(b, 49, 2048)).mean(dim=1)
        hmask = batch["history_mask"]
        denom = hmask.sum(dim=1, keepdim=True).clamp(min=1.0)
        hvis = (batch["history_vis"] * hmask[..., None]).sum(dim=1) / denom
        hpose = (batch["history_pose"] * hmask[..., None]).sum(dim=1) / denom
        hact = (batch["history_action"] * hmask[..., None]).sum(dim=1) / denom
        state = torch.cat([batch["belief"][:, BELIEF_IDX],
                           batch["match"][:, MATCH_IDX]], dim=-1)
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


class MediumPreset(nn.Module):
    def __init__(self, d=512, tok=36864, ffn_gc=4096, head_h=2048, n_heads=8):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(2048, tok), nn.GELU(), nn.Linear(tok, 2048))
        self.cur_proj = nn.Linear(2048, d)
        self.goal_proj = nn.Linear(512, d)
        self.goal_current_attn = nn.MultiheadAttention(
            d, n_heads, dropout=0.0, batch_first=True)
        self.gc_norm = nn.LayerNorm(d)
        self.gc_ffn = nn.Sequential(nn.Linear(d, ffn_gc), nn.GELU(),
                                    nn.Linear(ffn_gc, d))
        self.gc_ffn_norm = nn.LayerNorm(d)
        self.hist_vis_proj = nn.Sequential(nn.Linear(2048, d), nn.GELU())
        self.hist_pose_proj = nn.Sequential(nn.Linear(3, d), nn.GELU())
        self.hist_act_proj = nn.Sequential(nn.Linear(4, d), nn.GELU())
        self.state_proj = nn.Sequential(nn.Linear(12, d), nn.GELU())
        self.lm_proj = nn.Sequential(nn.Linear(3, d), nn.GELU())
        self.rel_proj = nn.Sequential(nn.Linear(3, d), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(7 * d, head_h), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(head_h, 4))

    def forward(self, batch):
        b = batch["current"].shape[0]
        cur = self.cur_proj(self.token_mlp(
            batch["current"].reshape(b, 49, 2048)))
        goal = self.goal_proj(batch["target_emb"])[:, None, :]
        matched, _ = self.goal_current_attn(query=goal, key=cur, value=cur)
        matched = self.gc_norm(goal + matched)
        matched = self.gc_ffn_norm(matched + self.gc_ffn(matched))
        hmask = batch["history_mask"]
        denom = hmask.sum(dim=1, keepdim=True).clamp(min=1.0)
        hvis = (batch["history_vis"] * hmask[..., None]).sum(dim=1) / denom
        hpose = (batch["history_pose"] * hmask[..., None]).sum(dim=1) / denom
        hact = (batch["history_action"] * hmask[..., None]).sum(dim=1) / denom
        state = torch.cat([batch["belief"][:, BELIEF_IDX],
                           batch["match"][:, MATCH_IDX]], dim=-1)
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


class TransformerPreset(nn.Module):
    """104-token explicit Transformer with 16 independent history tokens."""

    def __init__(self, d, layers, heads, ffn_gc, ffn_fusion):
        super().__init__()
        self.visual_proj = nn.Linear(2048, d)
        self.goal_proj = nn.Linear(512, d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.type_emb = nn.Embedding(7, d)
        self.current_pos = nn.Parameter(torch.zeros(1, 49, d))
        self.hist_time = nn.Parameter(torch.zeros(16, d))
        self.goal_current_attn = nn.MultiheadAttention(
            d, heads, dropout=DROPOUT, batch_first=True)
        self.gc_norm = nn.LayerNorm(d)
        self.gc_ffn = nn.Sequential(nn.Linear(d, ffn_gc), nn.GELU(),
                                    nn.Dropout(DROPOUT),
                                    nn.Linear(ffn_gc, d))
        self.gc_ffn_norm = nn.LayerNorm(d)
        self.hist_vis_proj = nn.Sequential(nn.Linear(2048, d), nn.GELU())
        self.hist_pose_proj = nn.Sequential(nn.Linear(3, d), nn.GELU())
        self.hist_act_proj = nn.Sequential(nn.Linear(4, d), nn.GELU())
        self.state_proj = nn.Sequential(nn.Linear(12, d), nn.GELU())
        self.lm_proj = nn.Sequential(nn.Linear(3, d), nn.GELU())
        self.rel_proj = nn.Sequential(nn.Linear(3, d), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ffn_fusion,
            dropout=DROPOUT, activation="gelu", batch_first=True,
            norm_first=True)
        self.fusion_encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.final_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 4)

    def forward(self, batch):
        b = batch["current"].shape[0]
        cur = (self.visual_proj(batch["current"].reshape(b, 49, 2048))
               + self.current_pos + self.type_emb.weight[1])
        goal = (self.goal_proj(batch["target_emb"])
                + self.type_emb.weight[2])[:, None, :]
        matched, _ = self.goal_current_attn(query=goal, key=cur, value=cur)
        matched = self.gc_norm(goal + matched)
        matched = self.gc_ffn_norm(matched + self.gc_ffn(matched))
        hist = (self.hist_vis_proj(batch["history_vis"])
                + self.hist_pose_proj(batch["history_pose"])
                + self.hist_act_proj(batch["history_action"])
                + self.hist_time[None] + self.type_emb.weight[3])
        state = torch.cat([batch["belief"][:, BELIEF_IDX],
                           batch["match"][:, MATCH_IDX]], dim=-1)
        state = (self.state_proj(state) + self.type_emb.weight[4])[:, None, :]
        lm_feats, rel, lm_mask, rel_mask = \
            TransformerM12Fix.landmark_geometry(batch["landmarks"],
                                                batch["states"])
        lm_tok = self.lm_proj(lm_feats) + self.type_emb.weight[5]
        rel_tok = self.rel_proj(rel) + self.type_emb.weight[6]
        cls_tok = self.cls_token.expand(b, -1, -1) + self.type_emb.weight[0]
        seq = torch.cat([cls_tok, cur, matched, hist, state, lm_tok, rel_tok],
                        dim=1)
        # 104-token mask: 1 cls + 49 current + 1 goal (never padded),
        # 16 history (padded exactly like TransformerM12Fix/V5), 1 state,
        # 8 landmark, 28 relation.  The history mask must not be dropped:
        # ignoring it would let the encoder attend to zero-filled slots.
        prefix = torch.zeros(b, 1 + 49 + 1, dtype=torch.bool,
                             device=lm_mask.device)
        hist_pad = batch["history_mask"] < 0.5
        mid = torch.zeros(b, 1, dtype=torch.bool, device=lm_mask.device)
        pad = torch.cat([prefix, hist_pad, mid, lm_mask, rel_mask], dim=1)
        out = self.fusion_encoder(seq, src_key_padding_mask=pad)
        return self.head(self.final_norm(out[:, 0]))


def build_model(name: str) -> nn.Module:
    if name == "small":
        return SmallPreset()
    if name == "medium":
        return MediumPreset()
    if name == "ts":
        return TransformerPreset(1408, 8, 8, 3 * 1408, 5376)
    if name == "tm":
        return TransformerM12Fix(d_model=2048, n_layers=10, n_heads=16,
                                 ffn=7424, dropout=DROPOUT)
    if name == "large":
        return TransformerPreset(2560, 12, 16, 4 * 2560, 10240)
    raise ValueError(f"unknown model {name}")


MODEL_NAMES = {"small": "Small", "medium": "Medium", "ts": "Transformer-S",
               "tm": "Transformer-M", "large": "Large"}
