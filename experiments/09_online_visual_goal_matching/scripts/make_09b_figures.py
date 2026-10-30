#!/usr/bin/env python3
"""09B figures: four-method SR, paired outcomes, update quality."""

from __future__ import annotations

import json
import math
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
STAGE08 = ROOT / "08_belief_only_geometric_navigation/results"
METHODS = ["transformer", "top1_static", "top5_static", "dynamic"]
LABEL = {"transformer": "Transformer-M", "top1_static": "Static Top-1",
         "top5_static": "Static Top-5", "dynamic": "Dynamic visual belief"}
COLOR = {"transformer": "#b23b3b", "top1_static": "#2f5d8a",
         "top5_static": "#4f9d69", "dynamic": "#7a3fa0"}


def load(path):
    return {json.loads(l)["episode_id"]: json.loads(l)
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def main() -> int:
    comp = json.loads((RES / "09b_comparison.json").read_text(encoding="utf-8"))
    d = load(RES / "09b_dynamic/journal.jsonl")
    print("computing update quality from episodes...")
    deltas, vis_errs, ents, ups = [], [], [], []
    for eid, r in d.items():
        rec = json.loads((RES / "09b_dynamic/episodes" /
                          f"{eid.replace(':', '_')}.json").read_text(encoding="utf-8"))
        target = np.asarray(rec["target"], float)
        prev = None
        for s in rec["trajectory"]:
            mp = np.asarray(s["map_candidate"], float)
            err = float(np.linalg.norm(mp - target))
            if s.get("confident"):
                vu = s["visual_offset_m"]
                vis = np.array([s["x"] + vu[0], s["y"] + vu[1]])
                vis_errs.append(float(np.linalg.norm(vis - target)))
            if s.get("updated") and prev is not None:
                deltas.append(err - prev)
            prev = err
        ents.append(r["mean_entropy"]); ups.append(r["belief_updates"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    m = comp["methods"]
    vals = [m[k]["sr"] for k in METHODS]
    los = [m[k]["sr"] - m[k]["sr_ci95"][0] for k in METHODS]
    his = [m[k]["sr_ci95"][1] - m[k]["sr"] for k in METHODS]
    ax.bar(range(4), vals, yerr=[los, his], capsize=4, color=[COLOR[k] for k in METHODS])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v * 100:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels([LABEL[k] for k in METHODS], rotation=15,
                                                fontsize=7)
    ax.set_ylim(0, 0.6); ax.set_ylabel("SR", fontsize=8)
    ax.set_title("SR: same 5,324 tasks (Wilson 95% CI)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    p = comp["paired"]["dynamic_vs_top1"]
    cats = [("both", "both succeed"), ("only_a", "only static Top-1"),
            ("only_b", "only dynamic"), ("neither", "both fail")]
    bottom = 0
    for key, name in cats:
        v = p[key] / p["n"]
        ax.bar([0], [v], 0.5, bottom=bottom, label=name)
        bottom += v
    ax.set_xticks([0]); ax.set_xticklabels(["dynamic vs static Top-1"], fontsize=8)
    ax.set_ylim(0, 1); ax.set_title("Paired outcome per task", fontsize=9)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    ax = axes[2]
    arr = np.array(deltas)
    changed = arr[np.abs(arr) > 1e-9]
    ax.hist(changed, bins=np.arange(-40, 41, 2), color=COLOR["dynamic"], alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("MAP error change after update (m; negative = better)", fontsize=8)
    ax.set_ylabel("updates (MAP changed)", fontsize=8)
    ax.set_title(f"Update quality: {len(changed)} changing updates of {len(arr)}; "
                 f"{100 * (changed < 0).mean():.0f}% improve", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "09b_sr_and_updates.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ve = np.array(vis_errs)
    axes[0].hist(ve, bins=np.arange(0, 101, 2), color="#2f5d8a")
    axes[0].axvline(20, color="#b23b3b", ls="--", lw=1)
    axes[0].set_xlabel("confident visual match error (m)", fontsize=8)
    axes[0].set_title(f"Visual match error at confident steps (n={len(ve)}; "
                      f">20 m: {(ve > 20).mean() * 100:.1f}%)", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].scatter(ents, ups, s=4, alpha=0.2, color="#7a3fa0")
    axes[1].set_xlabel("mean posterior entropy (normalised)", fontsize=8)
    axes[1].set_ylabel("updates per episode", fontsize=8)
    axes[1].set_title("Belief updates vs entropy", fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "09b_update_quality.png", dpi=150)
    plt.close(fig)
    print("figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
