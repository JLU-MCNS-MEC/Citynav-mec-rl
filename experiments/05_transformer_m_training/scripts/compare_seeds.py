#!/usr/bin/env python3
"""Compare the two Transformer-M seeds on the small-scale diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--figures", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    seeds = sorted(args.runs.glob("seed*/diagnosis.json"))
    if not seeds:
        raise SystemExit("no diagnosis.json found")
    data = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in seeds}

    rows = ["| seed | split | acc | macro-F1 | Stop R | TurnLeft R | TurnRight R | Forward R |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, d in data.items():
        for split in ("train", "val"):
            m = d[split]
            pc = m["per_class"]
            rows.append(
                f"| {name} | {split} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
                f"{pc['Stop']['recall']:.3f} | {pc['TurnLeft']['recall']:.3f} | "
                f"{pc['TurnRight']['recall']:.3f} | {pc['Forward']['recall']:.3f} |"
            )
    (args.out / "seed_comparison.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (args.out / "seed_comparison.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(CLASSES))
    for ax, split in zip(axes, ("train", "val")):
        for i, (name, d) in enumerate(data.items()):
            f1 = [d[split]["per_class"][c]["f1"] for c in CLASSES]
            ax.bar(x + (i - (len(data) - 1) / 2) * 0.35, f1, 0.35, label=name)
        ax.set_xticks(x); ax.set_xticklabels(CLASSES)
        ax.set_ylim(0, 1); ax.set_title(f"{split} per-class F1"); ax.legend()
    fig.tight_layout()
    fig.savefig(args.figures / "seed_comparison_f1.png", dpi=140)
    plt.close(fig)
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
