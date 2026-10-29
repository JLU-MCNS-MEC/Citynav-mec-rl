#!/usr/bin/env python3
"""Collect the audit Transformer-M arm results into CSV + markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

AUDIT = Path(__file__).resolve().parents[1]
CLASSES = ("Stop", "MoveForward", "TurnRight", "TurnLeft", "GoUp", "GoDown")


def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=AUDIT / "results")
    ap.add_argument("--out-csv", type=Path, default=AUDIT / "results/model_arms.csv")
    ap.add_argument("--out-md", type=Path, default=AUDIT / "results/model_arms.md")
    args = ap.parse_args()

    arms = []
    for path in sorted(args.results_dir.glob("train_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["results"]:
            split = row["final"]["val_unseen"]
            vs = row["final"]["val_seen"]
            arms.append({
                "arm": path.stem.replace("train_", ""),
                "seed": row["seed"],
                "vs_acc": vs["accuracy"],
                "vu_acc": split["accuracy"],
                "vu_move_acc": split["move_accuracy"],
                "vu_stop_p": split["stop_precision"],
                "vu_stop_r": split["stop_recall"],
                **{
                    f"vu_{name}": split["per_class_accuracy"].get(name, 0.0)
                    for name in CLASSES
                },
            })

    if not arms:
        raise SystemExit("no train_*.json results found")

    fields = list(arms[0].keys())
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(arms)

    by_arm = {}
    for row in arms:
        by_arm.setdefault(row["arm"], []).append(row)

    lines = [
        "| arm | n seeds | VU acc | VU move | VU Stop P/R | TurnRight | TurnLeft | GoUp | GoDown |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, rows in sorted(by_arm.items()):
        def ms(key):
            m, s = _mean_std([r[key] for r in rows])
            return f"{m:.3f}±{s:.3f}"
        lines.append(
            f"| {arm} | {len(rows)} | {ms('vu_acc')} | {ms('vu_move_acc')} | "
            f"{ms('vu_stop_p')} / {ms('vu_stop_r')} | {ms('vu_TurnRight')} | "
            f"{ms('vu_TurnLeft')} | {ms('vu_GoUp')} | {ms('vu_GoDown')} |"
        )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
