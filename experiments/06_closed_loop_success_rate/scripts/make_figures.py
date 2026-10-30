#!/usr/bin/env python3
"""Stage-06 figures: SR/OSR bars, action mix, distance histograms, trajectories.

Trajectories are drawn on the real north-up RGB raster with world coordinates;
nothing is synthesised.  English labels are used so the figures render without
a CJK font on the server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = HERE.parent / "results/seed42"
FIG = HERE.parent / "figures/seed42"
CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]


def load_rows(res_dir: Path) -> list[dict]:
    return [json.loads(line) for line in
            (res_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def bar_with_ci(ax, labels, values, cis, title, ylabel="SR"):
    xs = np.arange(len(labels))
    lo = [v - c[0] for v, c in zip(values, cis)]
    hi = [c[1] - v for v, c in zip(values, cis)]
    ax.bar(xs, values, color="#3b6ea5", width=0.6)
    ax.errorbar(xs, values, yerr=[lo, hi], fmt="none", ecolor="#222", capsize=4, lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(RESULTS))
    args = parser.parse_args()
    res_dir = Path(args.results)
    FIG.mkdir(parents=True, exist_ok=True)
    summary = json.loads((res_dir / "summary.json").read_text(encoding="utf-8"))
    rows = load_rows(res_dir)
    scopes = summary["scopes"]

    # 1. SR by split, 2. by length group, 3. by cohort
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    splits = ["val_seen", "val_unseen"]
    bar_with_ci(axes[0], splits, [scopes[s]["sr"] for s in splits],
                [scopes[s]["sr_ci95"] for s in splits], "SR by split")
    groups = ["short", "medium", "long"]
    labels, vals, cis = [], [], []
    for s in splits:
        for g in groups:
            labels.append(f"{s}\n{g}")
            vals.append(scopes[f"{s}|{g}"]["sr"])
            cis.append(scopes[f"{s}|{g}"]["sr_ci95"])
    bar_with_ci(axes[1], labels, vals, cis, "SR by length group")
    labels, vals, cis = [], [], []
    for s in splits:
        for c in ("A", "B"):
            labels.append(f"{s}\nround {c}")
            vals.append(scopes[f"{s}|cohort_{c}"]["sr"])
            cis.append(scopes[f"{s}|cohort_{c}"]["sr_ci95"])
    bar_with_ci(axes[2], labels, vals, cis, "SR by cohort")
    fig.tight_layout()
    fig.savefig(FIG / "sr_by_split_group_cohort.png", dpi=150)
    plt.close(fig)

    # 4. OSR vs SR
    fig, ax = plt.subplots(figsize=(6, 3.6))
    xs = np.arange(len(splits))
    ax.bar(xs - 0.2, [scopes[s]["osr"] for s in splits], 0.4, label="OSR", color="#7fa8d4")
    ax.bar(xs + 0.2, [scopes[s]["sr"] for s in splits], 0.4, label="SR", color="#2f5d8a")
    ax.set_xticks(xs)
    ax.set_xticklabels(splits)
    ax.set_ylim(0, 1)
    ax.set_title("OSR vs SR (OSR = ever entered 20 m)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "osr_vs_sr.png", dpi=150)
    plt.close(fig)

    # 5. action distribution
    fig, ax = plt.subplots(figsize=(6, 3.6))
    shares = [scopes["all"]["action_share"][c] for c in CLASSES]
    ax.bar(CLASSES, shares, color=["#b23b3b", "#3b6ea5", "#4f9d69", "#c58a2e"])
    for i, v in enumerate(shares):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of executed actions", fontsize=8)
    ax.set_title("Executed action distribution (all tasks)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "action_distribution.png", dpi=150)
    plt.close(fig)

    # 6. STOP distance histogram, 7. final distance histogram
    stop_d = [r["stop_dist"] for r in rows if r.get("stopped") and r.get("stop_dist") is not None]
    final_d = [r["final_dist"] for r in rows if r.get("final_dist") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].hist(stop_d, bins=np.arange(0, 405, 10), color="#3b6ea5")
    axes[0].axvline(20, color="#b23b3b", ls="--", lw=1)
    axes[0].set_title(f"distance at predicted STOP (n={len(stop_d)})", fontsize=9)
    axes[0].set_xlabel("m", fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].hist(final_d, bins=np.arange(0, 405, 10), color="#4f9d69")
    axes[1].axvline(20, color="#b23b3b", ls="--", lw=1)
    axes[1].set_title(f"final target distance (n={len(final_d)})", fontsize=9)
    axes[1].set_xlabel("m", fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "distance_histograms.png", dpi=150)
    plt.close(fig)

    # 8. case trajectories
    import sys
    sys.path.insert(0, str(HERE))
    from dataset_v2 import RasterStore

    by_id = {r["episode_id"]: r for r in rows}
    picks = []
    def first(pred):
        for r in sorted(rows, key=lambda x: x["episode_id"]):
            if pred(r):
                return r
        return None
    picks.append(("success", first(lambda r: r.get("failure") == "success")))
    picks.append(("reached_but_not_stopped",
                  first(lambda r: r.get("failure") == "reached_but_not_stopped")))
    picks.append(("premature_stop", first(lambda r: r.get("failure") == "premature_stop")))
    picks.append(("never_stop", first(lambda r: r.get("failure") == "never_stop")))
    picks.append(("collision", first(lambda r: r.get("failure") == "collision")))
    picks.append(("out_of_bounds", first(lambda r: r.get("failure") == "out_of_bounds")))

    rasters = RasterStore()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (name, rec) in zip(axes.ravel(), picks):
        if rec is None:
            ax.axis("off")
            ax.set_title(f"{name}: none", fontsize=8)
            continue
        path = res_dir / "episodes" / f"{rec['episode_id'].replace(':', '_')}.json"
        full = json.loads(path.read_text(encoding="utf-8"))
        traj = full["trajectory"]
        raster = rasters.get(rec["map"])
        pts = [[t["x"], t["y"]] for t in traj] or [full["start"][:2]]
        pts = np.asarray(pts)
        target = np.asarray(full["target"])
        allpts = np.vstack([pts, target[None]])
        lo = allpts.min(axis=0) - 40
        hi = allpts.max(axis=0) + 40
        lo = np.maximum(lo, [raster.left, raster.bottom])
        hi = np.minimum(hi, [raster.right, raster.top])
        c0, r0 = raster.pixel(lo[0], hi[1])
        c1, r1 = raster.pixel(hi[0], lo[1])
        c0, r0 = max(int(c0), 0), max(int(r0), 0)
        c1, r1 = min(int(c1), raster.image.shape[1]), min(int(r1), raster.image.shape[0])
        crop = raster.image[r0:r1, c0:c1]
        left = raster.left + c0 * raster.res
        right = raster.left + c1 * raster.res
        top = raster.top - r0 * raster.res
        bottom = raster.top - r1 * raster.res
        ax.imshow(crop, extent=[left, right, bottom, top], origin="upper")
        ax.plot(pts[:, 0], pts[:, 1], "-o", color="#1f4e79", lw=1.2, ms=2)
        ax.plot(pts[0, 0], pts[0, 1], "o", color="#2e8b57", ms=8, label="start")
        ax.plot(target[0], target[1], "*", color="#b23b3b", ms=13, label="target")
        if full.get("stop_step") is not None:
            ax.plot(pts[-1, 0], pts[-1, 1], "X", color="#d9822b", ms=10, label="STOP")
        mi = full.get("min_dist_step")
        if mi is not None and 0 <= mi < len(traj):
            ax.plot(traj[mi]["x"], traj[mi]["y"], "o", mfc="none", mec="#c58a2e", ms=10,
                    label="closest")
        ax.set_title(f"{rec['episode_id']} | {name} | steps={rec['steps']} "
                     f"min={rec['min_dist']:.1f}m final={rec['final_dist']:.1f}m",
                     fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
    axes[0, 0].legend(fontsize=6, loc="lower right")
    fig.suptitle("Closed-loop trajectories on the real top-down maps", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "trajectory_cases.png", dpi=150)
    plt.close(fig)
    print(f"figures written to {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
