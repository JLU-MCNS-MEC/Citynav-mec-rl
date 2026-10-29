#!/usr/bin/env python3
"""Merge closed-loop summaries across splits/seeds and draw the SR figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
SPLITS = ("val_seen", "val_unseen")
SEEDS = (42, 43)
BINS = ("short", "medium", "long")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--figures", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    data = {}
    for split in SPLITS:
        for seed in SEEDS:
            path = args.results / f"summary_{split}_seed{seed}.json"
            if path.exists():
                data[f"{split}/seed{seed}"] = json.loads(path.read_text(encoding="utf-8"))

    rows = ["| split | seed | group | n | SR | OSR | stop | early stop | non-stop | cap fail | steps | final dist | min dist |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, d in data.items():
        split, seed = name.split("/")
        for group in ("all",) + BINS:
            m = d["metrics"][group]
            if not m.get("n"):
                continue
            rows.append(
                f"| {split} | {seed} | {group} | {m['n']} | {m['sr']:.3f} | {m['osr']:.3f} | "
                f"{m['stop_rate']:.3f} | {m['early_stop_rate']:.3f} | {m['non_stop_rate']:.3f} | "
                f"{m['cap_fail_rate']:.3f} | {m['mean_steps']:.1f} | {m['mean_final_dist']:.1f} | "
                f"{m['mean_min_dist']:.1f} |"
            )
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "closed_loop_summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (args.results / "closed_loop_summary.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels, values = [], []
    for name, d in data.items():
        for group in BINS:
            m = d["metrics"][group]
            if m.get("n"):
                labels.append(f"{name.split('/')[0].replace('val_', '')}-{group}-{name.split('seed')[1]}")
                values.append(m["sr"])
    x = np.arange(len(labels))
    ax.bar(x, values, color="tab:blue")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("SR"); ax.set_ylim(0, 1)
    ax.set_title("closed-loop success rate by split / length bin / seed")
    fig.tight_layout()
    fig.savefig(args.figures / "closed_loop_sr.png", dpi=140)
    plt.close(fig)
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
