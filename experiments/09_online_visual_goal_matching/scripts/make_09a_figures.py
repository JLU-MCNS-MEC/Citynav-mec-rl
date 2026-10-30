#!/usr/bin/env python3
"""09A figures: localisation error, confidence, distance stratification, examples."""

from __future__ import annotations

import csv
import gzip
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "04_visual_goal_global_map_pipeline/scripts"))
sys.path.insert(0, str(ROOT / "05_transformer_m_training/scripts"))
import geom  # noqa: E402
import matching  # noqa: E402
from dataset_v2 import RasterStore  # noqa: E402

BINS = ["0-20", "20-40", "40-60", "60-80", ">80"]
METHODS = ("rgb_ncc", "layer3", "layer2")
LABEL = {"rgb_ncc": "RGB NCC", "layer3": "ResNet L3", "layer2": "ResNet L2"}
COLOR = {"rgb_ncc": "#b07b2b", "layer3": "#4f9d69", "layer2": "#2f5d8a"}


def load_results():
    with gzip.open(RES / "09a_match_results.csv.gz", "rt", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["in_view"] = int(r["in_view"])
        for m in METHODS:
            r[f"{m}_err_m"] = float(r[f"{m}_err_m"])
            r[f"{m}_score"] = float(r[f"{m}_score"])
    return rows


def main() -> int:
    rows = load_results()
    FIG.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for m in METHODS:
        err = np.array([r[f"{m}_err_m"] for r in rows if r["in_view"]],
                       float)
        err = err[np.isfinite(err)]
        xs = np.sort(err)
        ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), label=LABEL[m], color=COLOR[m])
    ax.set_xlim(0, 80); ax.set_xlabel("localisation error (m)", fontsize=8)
    ax.set_ylabel("fraction of states", fontsize=8)
    ax.axvline(10, color="k", ls="--", lw=1)
    ax.set_title("Error CDF (target inside the view)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    width = 0.26
    for j, m in enumerate(METHODS):
        med = []
        for b in BINS:
            e = np.array([r[f"{m}_err_m"] for r in rows
                          if r["in_view"] and r["distance_bin"] == b], float)
            e = e[np.isfinite(e)]
            med.append(float(np.median(e)) if len(e) else np.nan)
        ax.bar(np.arange(5) + (j - 1) * width, med, width, color=COLOR[m], label=LABEL[m])
    ax.set_xticks(range(5)); ax.set_xticklabels(BINS, fontsize=7)
    ax.set_xlabel("true distance bin (m)", fontsize=8)
    ax.set_ylabel("median error (m)", fontsize=8)
    ax.set_title("Median localisation error by distance", fontsize=9)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "09a_localization_error.png", dpi=150)
    plt.close(fig)

    # confidence: correct (in view, <=10 m) vs shuffled/wrong goals if available
    neg_path = RES / "09a_negative_per_state.csv.gz"
    neg = []
    if neg_path.exists():
        with gzip.open(neg_path, "rt", encoding="utf-8") as fh:
            neg = list(csv.DictReader(fh))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharex=True)
    for ax, m in zip(axes, METHODS):
        good = np.array([r[f"{m}_score"] for r in rows
                         if r["in_view"] and np.isfinite(r[f"{m}_err_m"])
                         and r[f"{m}_err_m"] <= 10], float)
        ax.hist(good, bins=np.linspace(0, 1, 40), alpha=0.6, color=COLOR[m],
                label="correct goal", density=True)
        if neg:
            for ntype, color in (("same_map_near", "#b23b3b"), ("shuffled", "#777777")):
                vals = np.array([float(r["score"]) for r in neg
                                 if r["method"] == m and r["type"] == ntype], float)
                if len(vals):
                    ax.hist(vals, bins=np.linspace(0, 1, 40), alpha=0.35, color=color,
                            label=f"wrong: {ntype.replace('_', ' ')}", density=True)
        ax.set_title(LABEL[m], fontsize=9); ax.set_xlabel("match score", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("density", fontsize=8)
    axes[0].legend(fontsize=6)
    fig.suptitle("Match confidence: correct goal vs wrong goals", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "09a_match_confidence.png", dpi=150)
    plt.close(fig)

    # distance stratification: precision@10, precision@20, false positives out of view
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, m in zip(axes, METHODS):
        p10, p20, fp = [], [], []
        for b in BINS:
            sel = [r for r in rows if r["distance_bin"] == b]
            inview = [r for r in sel if r["in_view"]]
            e = np.array([r[f"{m}_err_m"] for r in inview], float)
            p10.append(float(np.mean(e <= 10)) if len(e) else np.nan)
            p20.append(float(np.mean(e <= 20)) if len(e) else np.nan)
            out = [r for r in sel if not r["in_view"]]
            fp.append(float(np.mean([r[f"{m}_score"] >= 0.9 for r in out])) if out else np.nan)
        xs = np.arange(5)
        ax.bar(xs - 0.2, p10, 0.4, label="precision@10", color=COLOR[m])
        ax.bar(xs + 0.2, p20, 0.4, label="precision@20", color="#cccccc")
        ax.plot(xs, fp, "o-", color="#b23b3b", label="high-score rate when out of view")
        ax.set_xticks(xs); ax.set_xticklabels(BINS, fontsize=7)
        ax.set_title(LABEL[m], fontsize=9); ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("rate", fontsize=8)
    axes[0].legend(fontsize=6)
    fig.suptitle("Localisation by true distance", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "09a_distance_stratification.png", dpi=150)
    plt.close(fig)

    # examples: observation with true vs predicted markers (pose from the manifest)
    rasters = RasterStore()
    with open(RES / "09a_match_manifest.csv", encoding="utf-8") as fh:
        pose = {f"{r['episode_id']}:{r['step']}": r for r in csv.DictReader(fh)}
    picks = []
    for b in BINS:
        cand = [r for r in rows if r["in_view"] and r["distance_bin"] == b]
        if cand:
            picks.append(cand[len(cand) // 2])
    fig, axes = plt.subplots(1, len(picks), figsize=(3.1 * len(picks), 3.4))
    for ax, r in zip(np.atleast_1d(axes), picks):
        raster = rasters.get(r["map"])
        mrow = pose[r["state_id"]]
        obs = matching.prepare_observation(raster, float(mrow["x"]), float(mrow["y"]),
                                           float(mrow["yaw"]))
        ax.imshow(obs)
        tc, tr = float(r["true_col"]), float(r["true_row"])
        ax.plot(tc, tr, "o", mfc="none", mec="#2e8b57", ms=13, mew=2, label="true")
        for m, mk in (("layer2", "x"), ("layer3", "+")):
            ax.plot(float(r[f"{m}_col"]), float(r[f"{m}_row"]), mk, color=COLOR[m],
                    ms=8, mew=2, label=LABEL[m])
        ax.set_title(f"{r['episode_id']}\n{float(r['true_dist_m']):.0f} m", fontsize=7)
        ax.axis("off")
    axes0 = np.atleast_1d(axes)[0]
    axes0.legend(fontsize=6, loc="lower right")
    fig.suptitle("Matched target position in the north-up observation", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "09a_match_examples.png", dpi=150)
    plt.close(fig)
    print("figures written to", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
