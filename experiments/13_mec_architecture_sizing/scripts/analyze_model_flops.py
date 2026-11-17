#!/usr/bin/env python3
"""Stage 13: exact module-level FLOPs audit of the five Stage-11 experts.

Read-only.  Uses forward hooks on every nn.Linear and nn.MultiheadAttention
(2 x MACs).  The TransformerEncoderLayer fast path is replaced by its slow
Python implementation so the per-module hooks actually fire.  LayerNorm, GELU,
Softmax and other elementwise ops are not counted (negligible at GFLOPs scale).

Outputs: results/baseline_flops.json, results/module_flops_breakdown.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "experiments/11_teacher_geometry_ablation"
                       "/11B_transformer_m_v4/scripts"))

from model11_small import SmallMEC4  # noqa: E402
from model11_medium import MediumMEC4  # noqa: E402
from model11_explicit import TransformerS4, TransformerL4  # noqa: E402
from model11c_fix import TransformerM12Fix  # noqa: E402

MODELS = {
    "Small": SmallMEC4,
    "Medium": MediumMEC4,
    "Transformer-S": TransformerS4,
    "Transformer-M": TransformerM12Fix,
    "Large": TransformerL4,
}
TARGET = {
    "Small": 5e9,
    "Medium": 15e9,
    "Transformer-S": 40e9,
    "Transformer-M": 100e9,
    "Large": 200e9,
}


def make_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    b = 1
    hist_act = torch.zeros(b, 16, 4)
    hist_act[..., 1] = 1.0
    return {
        "current": torch.randn(b, 49, 2048, generator=g),
        "target_emb": torch.randn(b, 512, generator=g),
        "history_vis": torch.randn(b, 16, 2048, generator=g),
        "history_pose": torch.randn(b, 16, 3, generator=g),
        "history_action": hist_act,
        "history_mask": torch.ones(b, 16),
        "belief": torch.randn(b, 10, generator=g),
        "match": torch.randn(b, 5, generator=g),
        "states": torch.zeros(b, 3),
        "landmarks": torch.randn(b, 8, 3, generator=g),
    }


def _slow_encoder_layer_forward(self, src, src_mask=None,
                                src_key_padding_mask=None, is_causal=False):
    x = src
    if self.norm_first:
        x = x + self._sa_block(self.norm1(x), src_mask,
                               src_key_padding_mask, is_causal)
        x = x + self._ff_block(self.norm2(x))
    else:
        x = self.norm1(x + self._sa_block(x, src_mask,
                                          src_key_padding_mask, is_causal))
        x = self.norm2(x + self._ff_block(x))
    return x


def count_flops(model, batch):
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
                macs = x.numel() * module.out_features
                add(name, macs)
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
                handles.append(m.register_forward_hook(linear_hook(name), with_kwargs=True))
            elif isinstance(m, nn.MultiheadAttention):
                handles.append(m.register_forward_hook(mha_hook(name), with_kwargs=True))
        with torch.no_grad():
            out = model(batch)
        for h in handles:
            h.remove()
        return sum(counts.values()), counts, out
    finally:
        TransformerEncoderLayer.forward = old_forward


def main() -> int:
    (EXP / "results").mkdir(parents=True, exist_ok=True)
    batch = make_batch()
    rows = []
    summary = {}
    for name, cls in MODELS.items():
        model = cls().eval()
        params = sum(p.numel() for p in model.parameters())
        total, counts, out = count_flops(model, batch)
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:15]
        summary[name] = {
            "params": params,
            "flops": total,
            "flops_giga": total / 1e9,
            "target_gflops": TARGET[name] / 1e9,
            "ratio_target_over_current": TARGET[name] / max(total, 1),
            "output_shape": list(out.shape),
            "top_modules": top,
            "counter": ("forward hooks on Linear/MultiheadAttention, 2x MACs; "
                        "LayerNorm/GELU/Softmax not counted"),
        }
        for mod, fl in counts.items():
            rows.append({"model": name, "module": mod, "flops": int(fl),
                         "share": float(fl / total) if total else 0.0})
        print(f"{name:14s} params={params:10,d} "
              f"FLOPs={total/1e9:9.5f} GFLOPs "
              f"target={TARGET[name]/1e9:6.1f} "
              f"ratio={TARGET[name]/max(total,1):9.1f}x")
    (EXP / "results" / "baseline_flops.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    with (EXP / "results" / "module_flops_breakdown.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "module", "flops", "share"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {EXP/'results'/'baseline_flops.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
