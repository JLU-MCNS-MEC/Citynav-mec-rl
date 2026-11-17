#!/usr/bin/env python3
"""Stage 13-R3: hardware feasibility benchmark for the R2 preset models.

Read-only w.r.t. the repository models and checkpoints: it builds benchmark-only
preset classes with random weights, runs random-input forward / forward+backward,
times with CUDA events and records peak allocated memory.  No dataset, no
checkpoint, no optimizer step, no training, no closed-loop evaluation.

Usage:
  python benchmark_memory_latency.py --models all --batches 1 16 32 64 128
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
HF = HERE.parent
REV = HF.parent
EXP = REV.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "experiments/11_teacher_geometry_ablation"
                       "/11B_transformer_m_v4/scripts"))

from model11b import BELIEF_IDX, MATCH_IDX  # noqa: E402
from model11c_fix import TransformerM12Fix  # noqa: E402

DROPOUT = 0.1


def make_batch(b, device):
    return {
        "current": torch.randn(b, 49, 2048, device=device),
        "target_emb": torch.randn(b, 512, device=device),
        "history_vis": torch.randn(b, 16, 2048, device=device),
        "history_pose": torch.randn(b, 16, 3, device=device),
        "history_action": torch.randn(b, 16, 4, device=device),
        "history_mask": torch.ones(b, 16, device=device),
        "belief": torch.randn(b, 10, device=device),
        "match": torch.randn(b, 5, device=device),
        "states": torch.zeros(b, 3, device=device),
        "landmarks": torch.randn(b, 8, 3, device=device),
    }


class SmallPreset(nn.Module):
    """Small MLP + per-token visual MLP 2048->12288->2048, D=512, fusion 2048."""

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


class MediumPreset(nn.Module):
    """Medium cross-attention + per-token MLP 2048->36864->2048."""

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


class TransformerPreset(nn.Module):
    """104-token explicit Transformer with 16 history tokens."""

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
        pad = torch.cat([torch.zeros(b, 68, dtype=torch.bool,
                                     device=lm_mask.device),
                         lm_mask, rel_mask], dim=1)
        out = self.fusion_encoder(seq, src_key_padding_mask=pad)
        return self.head(self.final_norm(out[:, 0]))


def build_model(name):
    if name == "Small":
        return SmallPreset()
    if name == "Medium":
        return MediumPreset()
    if name == "Transformer-S":
        return TransformerPreset(1408, 8, 8, 3 * 1408, 5376)
    if name == "Transformer-M":
        return TransformerM12Fix(d_model=2048, n_layers=10, n_heads=16,
                                 ffn=7424, dropout=DROPOUT)
    if name == "Large":
        return TransformerPreset(2560, 12, 16, 4 * 2560, 10240)
    raise ValueError(name)


def measure(model, batch, mode, reps, warmup=2):
    """mode: 'forward' or 'fwd_bwd'."""
    device = next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    for _ in range(warmup):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
        if mode == "fwd_bwd":
            loss = out.float().mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e6
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(reps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
        if mode == "fwd_bwd":
            loss = out.float().mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / reps
    del out
    if mode == "fwd_bwd":
        del loss
    return ms, peak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--batches", nargs="+", type=int,
                    default=[1, 16, 32, 64, 128])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--backward-reps", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(HF / "results" / "benchmark_results.csv"))
    a = ap.parse_args()
    names = ["Small", "Medium", "Transformer-S", "Transformer-M", "Large"] \
        if a.models == ["all"] else a.models
    device = a.device
    rows = []
    for name in names:
        print(f"=== {name} ===", flush=True)
        model = build_model(name).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        rows.append({"model": name, "batch": 0, "params": n_params,
                     "kind": "params"})
        for b in a.batches:
            batch = make_batch(b, device)
            # forward latency (eval)
            model.eval()
            try:
                fwd_ms, fwd_peak = measure(model, batch, "forward", a.reps)
                fwd_status = "ok"
            except torch.cuda.OutOfMemoryError:
                fwd_ms, fwd_peak, fwd_status = None, None, "OOM"
                torch.cuda.empty_cache()
            # forward+backward (train)
            model.train()
            try:
                fb_ms, fb_peak = measure(model, batch, "fwd_bwd",
                                         a.backward_reps)
                fb_status = "ok"
            except torch.cuda.OutOfMemoryError:
                fb_ms, fb_peak, fb_status = None, None, "OOM"
                torch.cuda.empty_cache()
            opt_mb = n_params * 8 / 1e6
            train_peak = (fb_peak + opt_mb + 512.0) if fb_peak else None
            rows.append({
                "model": name, "batch": b, "params": n_params,
                "weights_fp32_mb": n_params * 4 / 1e6,
                "weights_bf16_mb": n_params * 2 / 1e6,
                "forward_ms": fwd_ms,
                "forward_per_sample_ms": (fwd_ms / b if fwd_ms else None),
                "forward_peak_mb": fwd_peak,
                "fwd_bwd_ms": fb_ms,
                "fwd_bwd_peak_mb": fb_peak,
                "optimizer_states_mb": opt_mb,
                "est_train_peak_mb": train_peak,
                "fits_32gb": (train_peak < 31000 if train_peak else None),
                "status": f"{fwd_status}/{fb_status}",
            })
            print(f"  B={b:3d} fwd={fwd_ms} ms peak={fwd_peak} MB | "
                  f"fwd+bwd={fb_ms} ms peak={fb_peak} MB | "
                  f"est_train={train_peak} MB | {fwd_status}/{fb_status}",
                  flush=True)
            del batch
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
        time.sleep(1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "batch", "params", "weights_fp32_mb", "weights_bf16_mb",
            "forward_ms", "forward_per_sample_ms", "forward_peak_mb",
            "fwd_bwd_ms", "fwd_bwd_peak_mb", "optimizer_states_mb",
            "est_train_peak_mb", "fits_32gb", "status"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        w.writerows([r for r in rows if r.get("kind") != "params"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
