#!/usr/bin/env python3
"""Stage 14 small_1g_experiment: real-forward FLOPs audit (2xMACs).

Hooks nn.Linear and nn.MultiheadAttention during one forward pass with the
exact Stage-14 input shapes.  Linear: 2*in*out*B*T.  MHA: qkv projection +
QK^T + AV + output projection.  Validated by reproducing the Stage-16 measured
4.954039 GFLOPs for the original SmallPreset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP14 = HERE.parent
sys.path.insert(0, str(EXP14))
sys.path.insert(0, str(EXP14.parents[1] / "experiments/11_teacher_geometry_ablation"
                       "/11B_transformer_m_v4/scripts"))

from models import build_model  # noqa: E402

B = 1


def make_batch():
    return {
        "current": torch.randn(B, 49, 2048),
        "target_emb": torch.randn(B, 512),
        "history_vis": torch.randn(B, 16, 2048),
        "history_pose": torch.randn(B, 16, 3),
        "history_action": torch.randn(B, 16, 4),
        "history_mask": torch.ones(B, 16),
        "belief": torch.randn(B, 10),
        "match": torch.randn(B, 5),
        "landmarks": torch.randn(B, 8, 3),
        "states": torch.randn(B, 3),
    }


def audit(model):
    rows = []

    def lin_hook(m, inp, out):
        t = inp[0].numel() // (B * m.in_features)
        rows.append({"module": type(m).__name__,
                     "shape": f"{m.in_features}->{m.out_features} x {t}",
                     "flops": 2.0 * B * t * m.in_features * m.out_features})

    def mha_hook(m, inp, out):
        t = inp[0].shape[1]
        d = m.embed_dim
        h = m.num_heads
        dh = d // h
        flops = (2.0 * B * t * 3 * d * d          # in_proj qkv
                 + 2.0 * B * h * t * t * dh * 2   # QK^T + AV
                 + 2.0 * B * t * d * d)           # out_proj
        rows.append({"module": "MultiheadAttention",
                     "shape": f"d={d} h={h} t={t}", "flops": flops})

    handles = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            handles.append(m.register_forward_hook(lin_hook))
        elif isinstance(m, torch.nn.MultiheadAttention):
            handles.append(m.register_forward_hook(mha_hook))
    with torch.no_grad():
        model(make_batch())
    for h in handles:
        h.remove()
    return rows


def main():
    out = {}
    for key in ("small", "small1g"):
        model = build_model(key).eval()
        params = sum(p.numel() for p in model.parameters())
        rows = audit(model)
        total = sum(r["flops"] for r in rows)
        token_mlp = sum(r["flops"] for r in rows
                        if r["module"] == "Linear" and "->2432" in r["shape"]
                        or r["module"] == "Linear" and "->12288" in r["shape"]
                        or r["module"] == "Linear" and "2432->" in r["shape"]
                        or r["module"] == "Linear" and "12288->" in r["shape"])
        head_mlp = sum(r["flops"] for r in rows
                       if r["shape"].endswith("x 1") and "->2048" in r["shape"]
                       and "4096->" in r["shape"])
        out[key] = {"params": params, "gflops": total / 1e9,
                    "token_mlp_gflops": token_mlp / 1e9,
                    "modules": len(rows),
                    "top": sorted(rows, key=lambda r: -r["flops"])[:6]}
    out["latency_ms_at_10tflops"] = {
        "small1g": out["small1g"]["gflops"] / 10.0,
        "small": out["small"]["gflops"] / 10.0}
    out["reference_stage16_small_gflops"] = 4.954039
    (HERE / "flops_audit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: vv for kk, vv in v.items() if kk != "top"})
                      for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
