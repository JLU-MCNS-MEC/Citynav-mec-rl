#!/usr/bin/env python3
"""Stage 08 figures: SR comparison, paired outcomes, behaviour, STOP error."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"
STAGE06 = ROOT / "06_closed_loop_success_rate/results/seed42"
METHODS = ["transformer", "top1", "top5"]
LABELS = {"transformer": "Transformer-M", "top1": "Top-1 geom", "top5": "Top-5 geom"}
COLORS = {"transformer": "#b23b3b", "top1": "#2f5d8a", "top5": "#4f9d69"}


def main() -> int:
    comp = json.loads((RES / "comparison.json").read_text(encoding="utf-8"))
    FIG.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    xs = np.arange(3)
    for i, split in enumerate(("val_seen", "val_unseen", "all")):
        block = comp["by_split"][split] if split != "all" else comp["methods"]
        vals, los, his = [], [], []
        for m in METHODS:
            v = block[m]
            vals.append(v["sr"])
            los.append(v["sr"] - v["sr_ci95"][0])
            his.append(v["sr_ci95"][1] - v["sr"])
        ax.bar(xs + (i - 1) * 0.25, vals, 0.24, yerr=[los, his], capsize=3,
               color=[COLORS[m] for m in METHODS], alpha=[0.55, 0.75, 1.0][i],
               label=["Val-Seen", "Val-Unseen", "All"][i])
    ax.set_xticks(xs); ax.set_xticklabels([LABELS[m] for m in METHODS], fontsize=8)
    ax.set_ylim(0, 0.6); ax.set_ylabel("SR", fontsize=8)
    ax.set_title("Success rate (paired tasks, Wilson 95% CI)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    groups = ["short", "medium", "long"]
    width = 0.26
    for j, m in enumerate(METHODS):
        vals = [comp["by_group"][f"val_seen|{g}"][m]["sr"] for g in groups]
        ax.bar(np.arange(3) + (j - 1) * width, vals, width, color=COLORS[m], label=LABELS[m])
    ax.set_xticks(range(3)); ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylim(0, 0.6); ax.set_title("SR by length group (Val-Seen)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    p = comp["paired"]["transformer_vs_top1"]
    p5 = comp["paired"]["transformer_vs_top5"]
    cats = ["both", "only_a", "only_b", "neither"]
    names = ["both succeed", "only Transformer", "only geometry", "both fail"]
    bottom = np.zeros(2)
    for k, (c, n) in enumerate(zip(cats, names)):
        vals = np.array([p[c], p5[c]]) / p["n"]
        ax.bar([0, 1], vals, 0.55, bottom=bottom, label=n)
        bottom += vals
    ax.set_xticks([0, 1]); ax.set_xticklabels(["vs Top-1", "vs Top-5"], fontsize=8)
    ax.set_ylim(0, 1); ax.set_title("Paired outcomes per task", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "sr_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    ax = axes[0]
    cats = ["premature_stop", "reached_but_not_stopped", "never_stop", "out_of_bounds",
            "collision", "success"]
    names = ["premature STOP", "reached, no STOP", "step cap", "out of bounds",
             "collision", "success"]
    bottom = np.zeros(3)
    for c, n in zip(cats, names):
        vals = np.array([comp["behaviour"][m]["failures"].get(c, 0) for m in METHODS]) / 5324
        ax.bar([0, 1, 2], vals, 0.6, bottom=bottom, label=n)
        bottom += vals
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels([LABELS[m] for m in METHODS], fontsize=7)
    ax.set_ylim(0, 1); ax.set_title("Outcome composition", fontsize=9)
    ax.legend(fontsize=6, loc="lower right")

    ax = axes[1]
    for j, m in enumerate(METHODS):
        v = comp["behaviour"][m]
        ax.bar(np.arange(2) + (j - 1) * 0.26,
               [v["turn_share"], v["turn_rate_when_behind"]], 0.24,
               color=COLORS[m], label=LABELS[m])
    ax.set_xticks(range(2)); ax.set_xticklabels(["turn share\n(all steps)",
                                                 "turn rate when\ntarget behind"], fontsize=7)
    ax.set_ylim(0, 1); ax.set_title("Turning behaviour", fontsize=9)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    for m in ("top1", "top5"):
        errs = []
        for path in sorted((RES / m / "episodes").glob("*.json")):
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec.get("stop_dist") is None or not rec.get("trajectory"):
                continue
            est = rec["trajectory"][-1].get("est_distance_m")
            if est is not None:
                errs.append(est - rec["stop_dist"])
        ax.hist(errs, bins=np.arange(-60, 61, 5), alpha=0.6, color=COLORS[m],
                label=f"{LABELS[m]} (n={len(errs)})")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("estimated - true distance at STOP (m)", fontsize=8)
    ax.set_title("STOP decision error", fontsize=9)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "behaviour.png", dpi=150)
    plt.close(fig)
    print(f"wrote {FIG / 'sr_comparison.png'} and {FIG / 'behaviour.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
