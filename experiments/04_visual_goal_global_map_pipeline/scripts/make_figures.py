#!/usr/bin/env python3
"""Recall@K figures for the 48 m baseline and the 24 m grid comparison."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
FIG = HERE / "figures"


def load(tag: str, split: str) -> dict:
    return json.loads((RESULTS / f"{tag}_{split}.json").read_text(encoding="utf-8"))


def main() -> int:
    FIG.mkdir(exist_ok=True)
    tags = [("validation_v2", "48 m grid"), ("validation_v2_stride24", "24 m grid")]
    splits = ["val_seen", "val_unseen"]
    ks = [1, 5, 10]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, radius in zip(axes, (20, 24)):
        width = 0.2
        x = np.arange(len(splits))
        for gi, (tag, label) in enumerate(tags):
            means = []
            for split in splits:
                d = load(tag, split)["all"]
                means.append([d[f"recall@{k}_{radius}m"] for k in ks])
            means = np.asarray(means)
            for ki, k in enumerate(ks):
                ax.bar(x + (gi - 0.5) * 0.4 + (ki - 1) * width, means[:, ki],
                       width=width, label=f"{label} R@{k}")
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.set_ylim(0, 1)
        ax.set_title(f"Recall@{'{1,5,10}'} within {radius} m")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("recall")
    axes[1].legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "validation_recall_v2.png", dpi=150)
    plt.close(fig)

    # per-map recall on the 24 m grid (val_seen), sorted
    by_map = defaultdict(list)
    with (RESULTS / "validation_v2_stride24_val_seen.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["phase"] != "t5":
                continue
            by_map[row["map"]].append(float(row["recall@1_20m"]))
    names = sorted(by_map)
    values = [float(np.mean(by_map[n])) for n in names]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(names)), values, color="tab:green")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=75, fontsize=7, ha="right")
    ax.set_ylabel("Recall@1 within 20 m")
    ax.set_title("24 m grid, Val-Seen, per map")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "recall_by_map_stride24.png", dpi=150)
    plt.close(fig)
    print("figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
