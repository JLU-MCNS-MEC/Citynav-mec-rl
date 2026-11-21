#!/usr/bin/env python3
"""Stage 16-P0: final FLOPs audit of the five Stage-14 frozen experts.

Same counting convention as Stage 13 (forward hooks on every nn.Linear and
nn.MultiheadAttention, 2 x MACs; LayerNorm/GELU/Softmax not counted), but
applied to the FINAL Stage-14 checkpoints and architectures in models.py.

Theoretical inference latency assumes a homogeneous 10 TFLOPS MEC node:
    t_ms = GFLOPs / 10 (GFLOPs per second = TFLOPS)
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
EXP16 = HERE.parent
REPO = EXP16.parents[1]
EXP14 = REPO / "experiments/14_mec_expert_retraining"
sys.path.insert(0, str(EXP14))
sys.path.insert(0, str(REPO / "experiments/11_teacher_geometry_ablation"
                       "/11B_transformer_m_v4/scripts"))
sys.path.insert(0, str(REPO / "experiments/10_goal_query_mec_training"
                       "/10D_transformer_m_training/scripts"))

from models import build_model, MODEL_NAMES  # noqa: E402

KEYS = ["small", "medium", "ts", "tm", "large"]
CKPTS = {
    "small": EXP14 / "checkpoints/Small/best.pt",
    "medium": EXP14 / "checkpoints/Medium/best.pt",
    "ts": EXP14 / "stability_revision/runs/ts/seed42/checkpoints/best.pt",
    "tm": EXP14 / "stability_revision/runs/tm/seed42/checkpoints/best.pt",
    "large": EXP14 / "stability_revision/runs/large/seed42/checkpoints/best.pt",
}
TARGET_GFLOPS = {
    "small": 4.954039296,
    "medium": 14.980623360,
    "ts": 39.671126528,
    "tm": 100.025417728,
    "large": 199.723980800,
}
MEC_TFLOPS = 10.0
COVERAGE_M = 500.0
MODEL_ORDER = ["small", "medium", "ts", "tm", "large"]
DEPLOY = MODEL_ORDER * 2


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


def make_batch(seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    b = 1
    hist_act = torch.zeros(b, 16, 4)
    hist_act[..., 1] = 1.0
    batch = {
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
    return {k: v.to(device) for k, v in batch.items()}


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
                handles.append(m.register_forward_hook(linear_hook(name),
                                                       with_kwargs=True))
            elif isinstance(m, nn.MultiheadAttention):
                handles.append(m.register_forward_hook(mha_hook(name),
                                                       with_kwargs=True))
        with torch.no_grad():
            out = model(batch)
        for h in handles:
            h.remove()
        return sum(counts.values()), counts, out
    finally:
        TransformerEncoderLayer.forward = old_forward


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    HERE.mkdir(parents=True, exist_ok=True)
    rows, summary = [], {}
    for key in KEYS:
        model = build_model(key).to(device).eval()
        ck = CKPTS[key]
        sha = hashlib.sha256(ck.read_bytes()).hexdigest()
        st = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        params = sum(p.numel() for p in model.parameters())
        total, counts, out = count_flops(model, make_batch(device=device))
        gflops = total / 1e9
        target = TARGET_GFLOPS[key]
        latency_ms = gflops / MEC_TFLOPS
        rel_err = (gflops - target) / target
        summary[key] = {
            "model": MODEL_NAMES[key],
            "checkpoint": str(ck),
            "checkpoint_sha256": sha,
            "params": params,
            "flops": int(total),
            "gflops": gflops,
            "stage13_r2_target_gflops": target,
            "relative_error_vs_target": rel_err,
            "mec_tflops": MEC_TFLOPS,
            "theoretical_infer_ms": latency_ms,
            "output_shape": list(out.shape),
            "counter": ("forward hooks on Linear/MultiheadAttention, 2x MACs; "
                        "LayerNorm/GELU/Softmax not counted"),
            "top_modules": sorted(counts.items(), key=lambda kv: -kv[1])[:12],
        }
        for mod, fl in counts.items():
            rows.append({"model": MODEL_NAMES[key], "key": key, "module": mod,
                         "flops": int(fl), "share": fl / total if total else 0.0})
        print(f"{MODEL_NAMES[key]:14s} params={params:12,d} "
              f"FLOPs={gflops:10.5f} G  target={target:8.5f} G  "
              f"rel_err={rel_err*100:+7.3f}%  t@10TFLOPS={latency_ms:7.3f} ms",
              flush=True)
        del model
        torch.cuda.empty_cache()

    (HERE / "model_flops_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    with (HERE / "model_flops_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "key", "params", "gflops", "stage13_r2_target_gflops",
                    "relative_error", "mec_tflops", "theoretical_infer_ms",
                    "checkpoint_sha256"])
        for key in KEYS:
            s = summary[key]
            w.writerow([s["model"], key, s["params"], f"{s['gflops']:.6f}",
                        f"{s['stage13_r2_target_gflops']:.6f}",
                        f"{s['relative_error_vs_target']:.6f}",
                        s["mec_tflops"], f"{s['theoretical_infer_ms']:.6f}",
                        s["checkpoint_sha256"]])
    with (HERE / "module_flops_breakdown.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "key", "module", "flops", "share"],
                           lineterminator="\n")
        w.writeheader(); w.writerows(rows)

    # deployment table (5-model equal, homogeneous 10 TFLOPS, 10 nodes)
    with (HERE / "deployment_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["node_id", "model_key", "model", "mec_tflops",
                    "model_gflops", "theoretical_infer_ms", "coverage_radius_m"])
        for i, key in enumerate(DEPLOY):
            w.writerow([i, key, MODEL_NAMES[key], MEC_TFLOPS,
                        f"{summary[key]['gflops']:.6f}",
                        f"{summary[key]['theoretical_infer_ms']:.6f}", COVERAGE_M])

    print("wrote", HERE)


if __name__ == "__main__":
    main()
