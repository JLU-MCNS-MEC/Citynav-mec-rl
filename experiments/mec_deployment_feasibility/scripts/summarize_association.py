#!/usr/bin/env python3
"""阶段 7 汇总：3 配置 × 5 策略的 SR / E2E / 时延占比。

A/B 读 replay_association.py 的 JSON；C 读 run_association_loop.py 的 journal。
输出 results/assoc_summary.json + fig_assoc_sr.png + fig_assoc_e2e.png + 报告表。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"

STRATEGIES = ["random", "nearest", "max_rate", "min_latency", "best_quality"]
CONFIGS = {"A": "A_10xSmall", "B": "B_10xTransformerM", "C": "C_5Small_5TM"}
LABELS = {"A": "A: 10×Small", "B": "B: 10×TM", "C": "C: 5+5"}


def load_c(strategy: str) -> dict:
    j = RES / f"assoc_C_{strategy}" / "journal.jsonl"
    if not j.exists():  # 尚未合并时读 part00
        j = RES / f"assoc_C_{strategy}" / "journal.part00.jsonl"
    rows = [json.loads(l) for l in j.read_text().splitlines()]
    ok = [r for r in rows if r.get("execution_status") == "ok"]
    e2e = np.asarray([r["e2e_mean_ms"] for r in ok], np.float64)
    # 逐 step 分位数：从 episodes 读轨迹
    step_e2e, step_inf, step_comm = [], [], []
    for r in ok:
        ep = RES / f"assoc_C_{strategy}" / "episodes" / f"{r['episode_id'].replace(':', '_')}.json"
        d = json.loads(ep.read_text())
        for s in d.get("trajectory", []):
            step_e2e.append(s["e2e_ms"]); step_inf.append(s["inf_ms"])
            step_comm.append(s["comm_ms"])
    step_e2e = np.asarray(step_e2e); step_inf = np.asarray(step_inf)
    step_comm = np.asarray(step_comm)
    return {"config": "C", "strategy": strategy, "n_tasks": len(rows),
            "n_ok": len(ok), "n_exception": len(rows) - len(ok),
            "sr": float(np.mean([r["success"] for r in ok])) if ok else None,
            "e2e_mean_ms": float(step_e2e.mean()),
            "e2e_p50_ms": float(np.percentile(step_e2e, 50)),
            "e2e_p95_ms": float(np.percentile(step_e2e, 95)),
            "e2e_p99_ms": float(np.percentile(step_e2e, 99)),
            "inf_mean_ms": float(step_inf.mean()),
            "comm_mean_ms": float(step_comm.mean()),
            "inf_share_mean": float((step_inf / step_e2e).mean()),
            "comm_share_mean": float((step_comm / step_e2e).mean())}


def main() -> int:
    out = {"summary": {}}
    for cfg in ("A", "B"):
        for s in STRATEGIES:
            p = RES / f"assoc_{cfg}_{s}.json"
            d = json.loads(p.read_text())
            out["summary"][f"{cfg}/{s}"] = {
                k: d[k] for k in ("n_tasks", "sr", "e2e_mean_ms", "e2e_p50_ms",
                                  "e2e_p95_ms", "e2e_p99_ms", "inf_mean_ms",
                                  "comm_mean_ms", "inf_share_mean",
                                  "comm_share_mean")}
    for s in STRATEGIES:
        d = load_c(s)
        out["summary"][f"C/{s}"] = d
        print(f"C/{s}: SR {d['sr']:.4f} | E2E {d['e2e_mean_ms']:.1f}ms "
              f"P95 {d['e2e_p95_ms']:.1f} | inf share {d['inf_share_mean']:.4f}")
    (RES / "assoc_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))

    # ---- 图：SR ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(len(STRATEGIES))
    for i, cfg in enumerate(("A", "B", "C")):
        ys = [out["summary"][f"{cfg}/{s}"]["sr"] * 100 for s in STRATEGIES]
        ax.bar(xs + (i - 1) * 0.27, ys, width=0.27, label=LABELS[cfg])
    ax.set_xticks(xs); ax.set_xticklabels(STRATEGIES, rotation=12)
    ax.set_ylabel("Val-Seen SR (%)"); ax.set_ylim(0, 70)
    ax.set_title("Closed-loop SR by config × association strategy")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(RES / "fig_assoc_sr.png", dpi=150)

    # ---- 图：E2E ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, cfg in enumerate(("A", "B", "C")):
        ys = [out["summary"][f"{cfg}/{s}"]["e2e_mean_ms"] for s in STRATEGIES]
        ax.bar(xs + (i - 1) * 0.27, ys, width=0.27, label=LABELS[cfg])
    ax.set_xticks(xs); ax.set_xticklabels(STRATEGIES, rotation=12)
    ax.set_ylabel("E2E latency mean (ms)")
    ax.set_title("Closed-loop E2E latency by config × association strategy")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(RES / "fig_assoc_e2e.png", dpi=150)
    print("saved assoc_summary.json + 2 figs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
