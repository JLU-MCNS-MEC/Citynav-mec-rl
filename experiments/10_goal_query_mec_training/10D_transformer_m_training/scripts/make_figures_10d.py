#!/usr/bin/env python3
"""10D figures: training curves, confusion matrices, GPU utilisation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"
RUN = RES / "runs_10d" / "seed42"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    hist = [json.loads(l) for l in (RUN / "history.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    if not hist:
        raise SystemExit("no history")
    epochs = [h["epoch"] for h in hist]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(epochs, [h["train_loss"] for h in hist], "o-", label="train")
    axes[0, 0].plot(epochs, [h["val_seen"]["loss"] for h in hist], "s-", label="val")
    axes[0, 0].set_title("loss")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()
    axes[0, 1].plot(epochs, [h["train_acc"] for h in hist], "o-", label="train acc")
    axes[0, 1].plot(epochs, [h["val_seen"]["accuracy"] for h in hist], "s-", label="val acc")
    axes[0, 1].plot(epochs, [h["val_seen"]["macro_f1"] for h in hist], "^-", label="val Macro-F1")
    axes[0, 1].set_title("accuracy / Macro-F1")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()
    axes[1, 0].plot(epochs, [h["seconds"] for h in hist], "o-")
    axes[1, 0].set_title("seconds per epoch")
    axes[1, 0].grid(alpha=0.3)
    gpu = []
    for h in hist:
        g = h.get("gpu", {})
        gpu.append(g.get("gpu_util_pct", np.nan))
    axes[1, 1].plot(epochs, gpu, "o-")
    axes[1, 1].set_title("GPU utilisation (%)")
    axes[1, 1].grid(alpha=0.3)
    fig.suptitle("10D: Transformer-M v3 training curves (seed 42)")
    fig.tight_layout()
    fig.savefig(FIG / "10d_training_curves.png", dpi=150)
    plt.close(fig)

    # confusion matrix of the best epoch (final best metrics)
    best_ep = max(hist, key=lambda h: h["val_seen"]["macro_f1"])
    conf = np.asarray(best_ep["val_seen"]["confusion"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(conf, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(conf[i, j]), ha="center", va="center",
                    color="white" if conf[i, j] > conf.max() / 2 else "black")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Val-Seen confusion matrix, best epoch {best_ep['epoch']} "
                 f"(Macro-F1 {best_ep['val_seen']['macro_f1']:.3f})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "10d_confusion.png", dpi=150)
    plt.close(fig)
    print(f"figures written | best epoch {best_ep['epoch']} "
          f"macro_f1={best_ep['val_seen']['macro_f1']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
