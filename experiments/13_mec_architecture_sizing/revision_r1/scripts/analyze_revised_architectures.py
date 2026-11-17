#!/usr/bin/env python3
"""Stage 13-R1: unified input-contract FLOPs audit (read-only).

- All Transformer models must use the canonical 104-token sequence:
  CLS 1 + Current 49 + Target 1 + History 16 + State 1 + Landmark 8 + Relation 28.
- Transformer-S / Large originally pool 16 history steps into one token (89).
  This script measures their FLOPs with 16 independent history tokens (104),
  reusing the original projection/encoder modules; no model definition changes.
- Small / Medium already consume all modalities; their internal mean-pooling /
  cross-attention is allowed and their FLOPs are unchanged.

Outputs under revision_r1/results/:
  revised_baseline_flops.csv
  input_contract_audit.csv
  module_flops_breakdown.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REV = HERE.parent
EXP = REV.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "experiments/11_teacher_geometry_ablation"
                       "/11B_transformer_m_v4/scripts"))
sys.path.insert(0, str(REPO / "experiments/13_mec_architecture_sizing/scripts"))

from model11_small import SmallMEC4  # noqa: E402
from model11_medium import MediumMEC4  # noqa: E402
from model11_explicit import TransformerS4, TransformerL4  # noqa: E402
from model11c_fix import TransformerM12Fix  # noqa: E402
from model11b import BELIEF_IDX, MATCH_IDX  # noqa: E402
from analyze_model_flops import make_batch, _slow_encoder_layer_forward  # noqa: E402

MODELS = {
    "Small": SmallMEC4,
    "Medium": MediumMEC4,
    "Transformer-S": TransformerS4,
    "Transformer-M": TransformerM12Fix,
    "Large": TransformerL4,
}
ORIGINAL_FLOPs = {
    "Small": 4473344,
    "Medium": 68224512,
    "Transformer-S": 119122560,
    "Transformer-M": 774068224,
    "Large": 650997952,
}
NEW_PARAMS = {  # +16h hist_time +7h type_emb -3h (current/goal type + goal_pos)
    "Transformer-S": 20 * 192,
    "Large": 20 * 352,
}


def count_flops(model, forward_fn):
    from torch.nn.modules.transformer import TransformerEncoderLayer
    old_forward = TransformerEncoderLayer.forward
    TransformerEncoderLayer.forward = _slow_encoder_layer_forward
    try:
        out_proj_ids = {id(m.out_proj) for m in model.modules()
                        if isinstance(m, nn.MultiheadAttention)}
        counts: dict[str, int] = {}

        def add(name, macs):
            counts[name] = counts.get(name, 0) + int(2 * macs)

        def linear_hook(name):
            def hook(module, args, kwargs, out):
                x = args[0] if args else kwargs["input"]
                add(name, x.numel() * module.out_features)
            return hook

        def mha_hook(name):
            def hook(module, args, kwargs, out):
                q = args[0] if len(args) > 0 else kwargs["query"]
                k = args[1] if len(args) > 1 else kwargs.get("key", q)
                v = args[2] if len(args) > 2 else kwargs.get("value", k)
                q_len = q.shape[-2]
                kv_len = k.shape[-2]
                d = module.embed_dim
                heads = module.num_heads
                hd = d // heads
                if module._qkv_same_embed_dim:
                    macs = q_len * d * d + 2 * kv_len * d * d
                else:
                    macs = (q_len * d * module.kdim
                            + kv_len * d * module.kdim
                            + kv_len * d * module.vdim)
                macs += 2 * heads * q_len * kv_len * hd
                macs += q_len * d * d
                add(name, macs)
            return hook

        handles = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                if id(m) in out_proj_ids:
                    continue
                handles.append(m.register_forward_hook(linear_hook(name),
                                                       with_kwargs=True))
            elif isinstance(m, nn.MultiheadAttention):
                handles.append(m.register_forward_hook(mha_hook(name),
                                                       with_kwargs=True))
        with torch.no_grad():
            out = forward_fn(model)
        for h in handles:
            h.remove()
        return sum(counts.values()), counts, out
    finally:
        TransformerEncoderLayer.forward = old_forward


def forward_104(model, batch):
    """Canonical 104-token forward for Transformer-S / Large.

    Reuses the original projections and encoder; history is 16 independent
    tokens instead of one mean-pooled token.  hist_time / type_emb are new
    parameters but add no matmul FLOPs, so they are omitted here.
    """
    b = batch["current"].shape[0]
    h = model.hidden_dim
    cur = (model.visual_proj(batch["current"].reshape(b, 49, 2048))
           + model.current_pos + model.current_type)
    goal = (model.goal_proj(batch["target_emb"])[:, None, :]
            + model.goal_pos + model.goal_type)
    matched, _ = model.goal_current_attn(query=goal, key=cur, value=cur)
    matched = model.gc_norm(goal + matched)
    matched = model.gc_ffn_norm(matched + model.gc_ffn(matched))
    hist = (model.hist_vis_proj(batch["history_vis"])
            + model.hist_pose_proj(batch["history_pose"])
            + model.hist_act_proj(batch["history_action"]))
    state = torch.cat([batch["belief"][:, BELIEF_IDX],
                       batch["match"][:, MATCH_IDX]], dim=-1)
    state = model.state_proj(state)[:, None, :]
    lm_feats, rel_feats, lm_mask, rel_mask = \
        TransformerM12Fix.landmark_geometry(batch["landmarks"],
                                            batch["states"])
    lm_tok = model.lm_proj(lm_feats)
    rel_tok = model.rel_proj(rel_feats)
    cls_tok = model.cls_token.expand(b, -1, -1)
    seq = torch.cat([cls_tok, cur, matched, hist, state, lm_tok, rel_tok],
                    dim=1)
    assert seq.shape[1] == 104, seq.shape
    pad = torch.cat([torch.zeros(b, 68, dtype=torch.bool,
                                 device=lm_mask.device),
                     lm_mask, rel_mask], dim=1)
    out = model.fusion_encoder(seq, src_key_padding_mask=pad)
    return model.head(model.final_norm(out[:, 0]))


def main() -> int:
    (REV / "results").mkdir(parents=True, exist_ok=True)
    batch = make_batch()
    rows = []
    breakdown = []
    for name, cls in MODELS.items():
        model = cls().eval()
        p0 = sum(p.numel() for p in model.parameters())
        if name in ("Transformer-S", "Large"):
            f0, _, _ = count_flops(model, lambda m: m(batch))
            f1, counts, _ = count_flops(model, lambda m: forward_104(m, batch))
            p1 = p0 + NEW_PARAMS[name]
            tok0, tok1 = 89, 104
            trans = True
        else:
            f0, counts, _ = count_flops(model, lambda m: m(batch))
            f1 = f0
            p1 = p0
            tok0 = tok1 = (104 if name == "Transformer-M" else None)
            trans = name == "Transformer-M"
        rows.append({
            "model": name,
            "is_transformer": trans,
            "tokens_before": tok0 if tok0 is not None else "n/a",
            "tokens_after": tok1 if tok1 is not None else "n/a",
            "params_before": p0,
            "params_after": p1,
            "flops_before": f0,
            "flops_after": f1,
            "increase_pct": 100.0 * (f1 - f0) / f0,
        })
        for mod, fl in counts.items():
            breakdown.append({"model": name, "module": mod, "flops": int(fl),
                              "share": float(fl / f1) if f1 else 0.0})
        print(f"{name:14s} {f0/1e6:9.3f} -> {f1/1e6:9.3f} MFLOPs "
              f"({100*(f1-f0)/f0:+6.2f}%) params {p0:,} -> {p1:,}")
    # consistency with Stage 13 baseline
    for r in rows:
        exp0 = ORIGINAL_FLOPs[r["model"]]
        assert abs(r["flops_before"] - exp0) / exp0 < 0.02, (
            r["model"], r["flops_before"], exp0)
    with (REV / "results" / "revised_baseline_flops.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with (REV / "results" / "module_flops_breakdown.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "module", "flops", "share"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(breakdown)
    (REV / "results" / "revised_baseline_flops.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")

    # input contract audit
    contract = [
        {"model": "Small", "is_transformer": False,
         "original_tokens": "MLP (no token sequence)",
         "revised_tokens": "MLP (no token sequence)",
         "history_before": "16-step masked mean-pool -> 1 vector",
         "history_after": "16-step masked mean-pool -> 1 vector (allowed)",
         "current": "49x2048", "target": "512", "state": "12",
         "landmark": "8x3", "relation": "28x3",
         "v5fix_geometry": "yes (TransformerM12Fix.landmark_geometry)",
         "mask_semantics": "history_mask < 0.5 (True=pad)",
         "satisfies_unified_info": "yes"},
        {"model": "Medium", "is_transformer": False,
         "original_tokens": "1 goal query + 49 current KV (no encoder)",
         "revised_tokens": "1 goal query + 49 current KV (no encoder)",
         "history_before": "16-step masked mean-pool -> 1 vector",
         "history_after": "16-step masked mean-pool -> 1 vector (allowed)",
         "current": "49x2048", "target": "512", "state": "12",
         "landmark": "8x3", "relation": "28x3",
         "v5fix_geometry": "yes (TransformerM12Fix.landmark_geometry)",
         "mask_semantics": "history_mask < 0.5 (True=pad)",
         "satisfies_unified_info": "yes"},
        {"model": "Transformer-S", "is_transformer": True,
         "original_tokens": "89",
         "revised_tokens": "104",
         "history_before": "16-step mean-pool -> 1 token",
         "history_after": "16 independent history tokens + hist_time",
         "current": "49x2048", "target": "512", "state": "12",
         "landmark": "8x3", "relation": "28x3",
         "v5fix_geometry": "yes (TransformerM12Fix.landmark_geometry)",
         "mask_semantics": "history_mask < 0.5 (True=pad); prefix 68",
         "satisfies_unified_info": "yes after revision"},
        {"model": "Transformer-M", "is_transformer": True,
         "original_tokens": "104", "revised_tokens": "104",
         "history_before": "16 independent history tokens",
         "history_after": "16 independent history tokens",
         "current": "49x2048", "target": "512", "state": "12",
         "landmark": "8x3", "relation": "28x3",
         "v5fix_geometry": "yes (TransformerM12Fix.landmark_geometry)",
         "mask_semantics": "history_mask < 0.5 (True=pad); prefix 68",
         "satisfies_unified_info": "yes (canonical)"},
        {"model": "Large", "is_transformer": True,
         "original_tokens": "89", "revised_tokens": "104",
         "history_before": "16-step mean-pool -> 1 token",
         "history_after": "16 independent history tokens + hist_time",
         "current": "49x2048", "target": "512", "state": "12",
         "landmark": "8x3", "relation": "28x3",
         "v5fix_geometry": "yes (TransformerM12Fix.landmark_geometry)",
         "mask_semantics": "history_mask < 0.5 (True=pad); prefix 68",
         "satisfies_unified_info": "yes after revision"},
    ]
    with (REV / "results" / "input_contract_audit.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(contract[0].keys()),
                           lineterminator="\n")
        w.writeheader()
        w.writerows(contract)
    print(f"wrote {REV/'results'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
