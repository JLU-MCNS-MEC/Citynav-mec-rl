#!/usr/bin/env python3
"""阶段 4：单次决策 FLOPs 实测计数 + 计算能力敏感性分析。

- FLOPs 计数：运行期 hook（Linear: 2·in·out/个 token；MultiheadAttention 前向
  预钩子另加注意力矩阵乘 4·q·kv·d），与 mec-v8.2 latency_model 计数口径一致
  （2×MACs，不计 LayerNorm/GELU）。三个模型用同一个真实 batch 各跑一次。
- 理论时延 T_inf = FLOPs / MEC 算力，算力 ∈ {0.1,0.2,0.3,0.5,1.0} TFLOPS
  （算力为**人为建模假设**，非硬件实测——历史 0.2 TFLOPS 出处不可追溯，
  按用户指示做全档位扫描）。
- 与阶段 3 实测前向时延（inference_latency.json）对比，输出
  results/flops_capacity.json + 图表 PNG。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/11_teacher_geometry_ablation"
                       "/11B_transformer_m_v4/scripts"))

from model11_small import SmallMEC4  # noqa: E402
from model11_medium import MediumMEC4  # noqa: E402
from model11c_fix import TransformerM12Fix  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
CAPACITIES = [0.1e12, 0.2e12, 0.3e12, 0.5e12, 1.0e12]


class FlopCounter:
    def __init__(self):
        self.flops = 0
        self.handles = []

    def attach(self, module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                self.handles.append(m.register_forward_hook(self._lin))
            elif isinstance(m, nn.MultiheadAttention):
                self.handles.append(
                    m.register_forward_pre_hook(self._mha, with_kwargs=True))

    def _lin(self, mod, args, out):
        x = args[0]
        tok = x.numel() // x.shape[-1]  # 总 token 数（含 batch）
        self.flops += 2 * mod.in_features * mod.out_features * tok

    def _mha(self, mod, args, kwargs=None):
        if kwargs is None:
            kwargs = {}
        def get(name, idx):
            if idx < len(args) and args[idx] is not None:
                return args[idx]
            return kwargs.get(name)
        q = get("query", 0)
        k = get("key", 1)
        if q is None or k is None:
            return
        q_len, kv_len, d = q.shape[1], k.shape[1], q.shape[-1]
        self.flops += 4 * q_len * kv_len * d * q.shape[0]

    def reset(self):
        self.flops = 0

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def build_batch(device="cpu"):
    b = {
        "current": torch.randn(1, 49, 2048),
        "target_emb": torch.randn(1, 512),
        "history_vis": torch.randn(1, 16, 2048),
        "history_pose": torch.randn(1, 16, 3),
        "history_action": torch.randn(1, 16, 4),
        "history_mask": torch.ones(1, 16),
        "belief": torch.randn(1, 10),
        "match": torch.randn(1, 5),
        "landmarks": torch.randn(1, 8, 3),
        "states": torch.randn(1, 3),
    }
    return {k: v.to(device) for k, v in b.items()}


def main() -> int:
    torch.manual_seed(0)
    models = {"small": SmallMEC4, "medium": MediumMEC4,
              "transformer_m": TransformerM12Fix}
    batch = build_batch("cpu")
    flops = {}
    for name, cls in models.items():
        m = cls().eval()
        ctr = FlopCounter()
        ctr.attach(m)
        with torch.no_grad():
            m(batch)
        ctr.detach()
        flops[name] = int(ctr.flops)
        print(f"{name}: {ctr.flops/1e6:.1f} MFLOPs")

    lat = json.loads((RES / "inference_latency.json").read_text())
    measured_ms = {k: lat[k]["mean_ms"]
                   for k in ("small", "medium", "transformer_m")}

    out = {"flops": flops, "measured_mean_ms": measured_ms,
           "capacities_flops_per_s": CAPACITIES, "per_capacity": {}}
    for cap in CAPACITIES:
        row = {}
        for name, f in flops.items():
            t_ms = f / cap * 1e3
            ratio = t_ms / measured_ms[name]
            row[name] = {"t_inf_ms": t_ms, "ratio_to_measured": ratio}
        row["delta_small_tm_ms"] = row["transformer_m"]["t_inf_ms"] - row["small"]["t_inf_ms"]
        row["delta_medium_tm_ms"] = row["transformer_m"]["t_inf_ms"] - row["medium"]["t_inf_ms"]
        out["per_capacity"][f"{cap/1e12:.1f} TFLOPS"] = row

    (RES / "flops_capacity.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))

    # ---- 图 1: 理论时延 vs 算力 ----
    caps = [c / 1e12 for c in CAPACITIES]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, label in [("small", "Small"), ("medium", "Medium"),
                        ("transformer_m", "Transformer-M")]:
        ys = [out["per_capacity"][f"{c} TFLOPS"][name]["t_inf_ms"]
              for c in caps]
        ax.plot(caps, ys, marker="o", label=label)
    ax.set_xlabel("MEC 算力 (TFLOPS，人为建模假设)")
    ax.set_ylabel("理论推理时延 T_inf (ms)")
    ax.set_title("单次决策理论时延 vs MEC 算力")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "fig_theoretical_latency.png", dpi=150)

    # ---- 图 2: 实测 vs 理论（0.2 TFLOPS 参照线 + 全档位比值）----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(3)
    ax.bar(xs - 0.2, [measured_ms[n] for n in ("small", "medium",
                                               "transformer_m")],
            width=0.4, label="RTX 5090 实测 (mean)")
    ratios = [out["per_capacity"]["0.2 TFLOPS"][n]["ratio_to_measured"]
              for n in ("small", "medium", "transformer_m")]
    ax2 = ax.twinx()
    ax2.plot(xs, ratios, "r.--", label="理论/实测比值 @0.2 TFLOPS")
    ax.set_xticks(xs)
    ax.set_xticklabels(["Small", "Medium", "Transformer-M"])
    ax.set_ylabel("时延 (ms)")
    ax2.set_ylabel("理论/实测比值")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "fig_measured_vs_theoretical.png", dpi=150)
    print("saved results/flops_capacity.json + 2 张图")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
