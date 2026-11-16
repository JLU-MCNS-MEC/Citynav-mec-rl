#!/usr/bin/env python3
"""Stage 12 aggregation: tables, bar charts, MA100 convergence curves, Pareto.

Reads whatever runs exist and writes markdown/JSON under results/summary and
figures under results/analysis.  Safe to run while experiments are in progress.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GROUP_ORDER = ("two_homo", "five_homo", "two_hetero", "five_hetero")
STRATEGIES = ("random", "max_rate", "max_quality", "oracle")
ALGOS = ("ppo", "dqn")
SEEDS = (42, 43)
OUT = P.RES / "summary"
FIG = P.RES / "analysis"


def load_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def baseline_rows():
    rows = []
    for g in GROUP_ORDER:
        for s in STRATEGIES:
            d = load_json(P.RES / "baselines" / g / s / f"seed{P.EVAL_SEED}"
                          / "summary.json")
            if d:
                rows.append({"group": g, "policy": s, **d})
    return rows


def rl_rows():
    rows = []
    for g in GROUP_ORDER:
        for algo in ALGOS:
            for seed in SEEDS:
                d = load_json(P.RES / "eval" / f"{g}_{algo}_seed{seed}"
                              / "summary.json")
                if d:
                    rows.append({"group": g, "policy": f"{algo}-s{seed}", **d})
    return rows


def fmt(x, nd=2, pct=False):
    if x is None:
        return "-"
    if pct:
        return f"{100*x:.2f}%"
    return f"{x:.{nd}f}"


def table(rows, title):
    cols = ["group", "policy", "sr", "e2e_mean_ms", "e2e_p95_ms",
            "inf_mean_ms", "comm_mean_ms", "encode_mean_ms"]
    lines = [f"### {title}", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join([
            r["group"], r["policy"], fmt(r.get("sr"), pct=True),
            fmt(r.get("e2e_mean_ms")), fmt(r.get("e2e_p95_ms")),
            fmt(r.get("inf_mean_ms")), fmt(r.get("comm_mean_ms")),
            fmt(r.get("encode_mean_ms"))]) + " |")
    return "\n".join(lines) + "\n"


def share_table(rows, title):
    models = P.MODEL_LABEL.values()
    cols = ["group", "policy"] + list(models)
    lines = [f"### {title}", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        sh = r.get("model_step_share", {})
        vals = []
        for m in P.MODEL_ORDER:
            key = P.MODEL_LABEL[m]
            v = sh.get(key, sh.get(m, 0.0))
            vals.append(f"{100*float(v):.1f}%")
        lines.append("| " + " | ".join([r["group"], r["policy"]] + vals) + " |")
    return "\n".join(lines) + "\n"


def bar_figures(rows):
    if not rows:
        return
    FIG.mkdir(parents=True, exist_ok=True)
    for g in GROUP_ORDER:
        gr = [r for r in rows if r["group"] == g]
        if not gr:
            continue
        labels = [r["policy"] for r in gr]
        x = np.arange(len(labels))
        fig, axes = plt.subplots(1, 3, figsize=(max(9, 2.2 * len(labels)), 4.2))
        metrics = [("sr", "SR (%)", 100.0), ("e2e_mean_ms", "E2E mean (ms)", 1.0),
                   ("inf_mean_ms", "MEC inference (ms)", 1.0)]
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(labels)))
        for ax, (key, label, scale) in zip(axes, metrics):
            vals = [float(r.get(key, 0.0)) * scale for r in gr]
            ax.bar(x, vals, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=0.25)
            for xi, v in zip(x, vals):
                ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=7)
        fig.suptitle(f"{g} — {P.GROUP_SPECS[g]['label']}")
        fig.tight_layout()
        fig.savefig(FIG / f"bars_{g}.png", dpi=160)
        plt.close(fig)


def pareto_figure(rows):
    if not rows:
        return
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    markers = {"random": "o", "max_rate": "s", "max_quality": "^",
               "oracle": "*", "ppo-s42": "D", "ppo-s43": "d",
               "dqn-s42": "v", "dqn-s43": "P"}
    for g in GROUP_ORDER:
        gr = [r for r in rows if r["group"] == g]
        for r in gr:
            ax.scatter(r["e2e_mean_ms"], 100 * r["sr"],
                       marker=markers.get(r["policy"], "o"),
                       label=f"{g}/{r['policy']}", s=55)
    ax.set_xlabel("E2E mean (ms)")
    ax.set_ylabel("SR (%)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "pareto_sr_e2e.png", dpi=160)
    plt.close(fig)


def convergence_figures():
    FIG.mkdir(parents=True, exist_ok=True)
    for g in GROUP_ORDER:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        found = False
        for algo in ALGOS:
            for seed in SEEDS:
                hp = P.RES / "rl" / g / algo / f"seed{seed}" / "history.jsonl"
                if not hp.exists():
                    continue
                rows = [json.loads(l) for l in hp.read_text(
                    encoding="utf-8").splitlines() if l.strip()]
                if not rows:
                    continue
                found = True
                ep = [r["episode"] for r in rows]
                rew = [r.get("ma100_reward") for r in rows]
                e2e = [r.get("ma100_e2e_ms") for r in rows]
                axes[0].plot(ep, rew, label=f"{algo}-s{seed}", lw=1.2)
                axes[1].plot(ep, e2e, label=f"{algo}-s{seed}", lw=1.2)
        if not found:
            plt.close(fig)
            continue
        axes[0].set_xlabel("episode")
        axes[0].set_ylabel("MA100 reward (scheme D)")
        axes[0].set_title("Reward (100-episode moving average)")
        axes[1].set_xlabel("episode")
        axes[1].set_ylabel("MA100 E2E (ms)")
        axes[1].set_title("E2E latency (100-episode moving average)")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
        fig.suptitle(f"{g} — {P.GROUP_SPECS[g]['label']}")
        fig.tight_layout()
        fig.savefig(FIG / f"m100_{g}.png", dpi=160)
        plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    b = baseline_rows()
    r = rl_rows()
    all_rows = b + r
    parts = ["# Stage 12 — baseline and RL association results", ""]
    parts.append(table(b, "Baselines (Val-Seen 2,498)"))
    parts.append(table(r, "RL (Val-Seen 2,498)"))
    parts.append(share_table(all_rows, "Expert step share"))
    (OUT / "RESULTS.md").write_text("\n".join(parts), encoding="utf-8")
    (OUT / "results.json").write_text(
        json.dumps({"baselines": b, "rl": r}, indent=1), encoding="utf-8")
    bar_figures(all_rows)
    pareto_figure(all_rows)
    convergence_figures()
    print(f"wrote {OUT/'RESULTS.md'} ({len(b)} baseline rows, {len(r)} RL rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
