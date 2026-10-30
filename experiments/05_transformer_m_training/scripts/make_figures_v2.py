#!/usr/bin/env python3
"""Training curves and per-epoch tables for the stage-05 runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
RUNS = HERE / "runs_v2"
FIG = HERE / "figures"
RESULTS = HERE / "results"


def load_history(run: str) -> list[dict]:
    path = RUNS / run / "history.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["seed42", "seed43"])
    parser.add_argument("--tag", default="formal")
    args = parser.parse_args()
    FIG.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    histories = {r: load_history(r) for r in args.runs}
    histories = {k: v for k, v in histories.items() if v}
    if not histories:
        print("no history found")
        return 1

    rows = []
    for name, hist in histories.items():
        for h in hist:
            rows.append({"run": name, "epoch": h["epoch"], "train_loss": h["train_loss"],
                         "train_acc": h["train_acc"], "train_subset_acc": h["train_subset"]["accuracy"],
                         "train_subset_macro_f1": h["train_subset"]["macro_f1"],
                         "val_acc": h["val_seen"]["accuracy"],
                         "val_macro_f1": h["val_seen"]["macro_f1"],
                         "val_stop_recall": h["val_seen"]["per_class"]["Stop"]["recall"],
                         "val_turn_recall": 0.5 * (h["val_seen"]["per_class"]["TurnLeft"]["recall"]
                                                   + h["val_seen"]["per_class"]["TurnRight"]["recall"]),
                         "lr": h["lr"], "seconds": h["seconds"]})
    with (RESULTS / f"training_curves_{args.tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    panels = [("train_loss", "train loss"), ("train_acc", "train accuracy (running)"),
              ("val_acc", "val-seen accuracy"), ("val_macro_f1", "val-seen macro-F1"),
              ("val_stop_recall", "val-seen STOP recall"),
              ("val_turn_recall", "val-seen turn recall (L/R mean)")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for name in histories:
            xs = [h["epoch"] for h in histories[name]]
            ys = []
            for h in histories[name]:
                if key == "val_turn_recall":
                    ys.append(0.5 * (h["val_seen"]["per_class"]["TurnLeft"]["recall"]
                                     + h["val_seen"]["per_class"]["TurnRight"]["recall"]))
                else:
                    ys.append(h[key])
            ax.plot(xs, ys, label=name, lw=1.4)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Transformer-M training, fixed-50 m direct teacher", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"training_curves_{args.tag}.png", dpi=150)
    plt.close(fig)
    print("wrote", RESULTS / f"training_curves_{args.tag}.csv", "and figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
