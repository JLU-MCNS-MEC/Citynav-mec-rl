#!/usr/bin/env python3
"""Evaluate a trained Transformer-M and draw small-scale diagnostic figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from data import EpisodeStore  # noqa: E402
from model import N_ACTIONS, TransformerM  # noqa: E402
from train import CLASSES, evaluate  # noqa: E402

CANDIDATES = HERE.parent / "04_visual_goal_global_map_pipeline/cache/candidates"


def load_model(run: Path, device: str, which: str = "final"):
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    model = TransformerM(config["d_model"], config["layers"], config["heads"])
    if which == "best" and (run / "best.pt").exists():
        ckpt = run / "best.pt"
    else:
        ckpt = run / "checkpoint.pt" if (run / "checkpoint.pt").exists() else run / "best.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, config, ckpt.name


def plot_curves(run: Path, figures: Path) -> None:
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train loss")
    axes[0].set_title("loss"); axes[0].legend()
    axes[1].plot(epochs, [h["train_eval"]["accuracy"] for h in history], label="train acc")
    axes[1].plot(epochs, [h["val"]["accuracy"] for h in history], label="val acc")
    axes[1].set_title("accuracy"); axes[1].legend()
    for name in CLASSES:
        axes[2].plot(epochs, [h["val"]["per_class"][name]["recall"] for h in history], label=name)
    axes[2].set_title("val recall per class"); axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "training_curves.png", dpi=140)
    plt.close(fig)


def plot_confusion(metrics: dict, path: Path, title: str) -> None:
    conf = np.asarray(metrics["confusion"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(N_ACTIONS)); ax.set_yticks(range(N_ACTIONS))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right"); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(N_ACTIONS):
        for j in range(N_ACTIONS):
            ax.text(j, i, str(conf[i, j]), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=HERE / "features")
    parser.add_argument("--selection", type=Path, default=HERE / "results/selection.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ckpt", choices=("final", "best"), default="final")
    parser.add_argument("--figures", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    args.figures.mkdir(parents=True, exist_ok=True)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    store = EpisodeStore(args.features, CANDIDATES)
    model, config, ckpt_name = load_model(args.run, args.device, args.ckpt)

    train_metrics = evaluate(model, store, selection["train"], args.device, max_samples=100000, seed=0)
    val_metrics = evaluate(model, store, selection["val"], args.device, max_samples=100000, seed=1)
    result = {
        "checkpoint": ckpt_name,
        "n_params": config.get("n_params"),
        "train": train_metrics,
        "val": val_metrics,
        "overfit_gap_accuracy": train_metrics["accuracy"] - val_metrics["accuracy"],
    }
    (args.run / "diagnosis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot_curves(args.run, args.figures)
    plot_confusion(train_metrics, args.figures / f"{args.run.name}_confusion_train.png", "train")
    plot_confusion(val_metrics, args.figures / f"{args.run.name}_confusion_val.png", "val")

    rows = [f"# {args.run.name} diagnosis", "", f"- checkpoint: {ckpt_name}",
            f"- params: {config.get('n_params'):,}", ""]
    rows.append("| class | train P/R/F1 | val P/R/F1 | val support |")
    rows.append("|---|---|---|---:|")
    for name in CLASSES:
        t = train_metrics["per_class"][name]
        v = val_metrics["per_class"][name]
        rows.append(f"| {name} | {t['precision']:.3f}/{t['recall']:.3f}/{t['f1']:.3f} | "
                    f"{v['precision']:.3f}/{v['recall']:.3f}/{v['f1']:.3f} | {v['support']} |")
    rows.append("")
    rows.append(f"- train acc {train_metrics['accuracy']:.4f}, val acc {val_metrics['accuracy']:.4f}, "
                f"gap {result['overfit_gap_accuracy']:+.4f}")
    rows.append(f"- val macro-F1 {val_metrics['macro_f1']:.4f}")
    rows.append(f"- train predictions {train_metrics['prediction_counts']}")
    rows.append(f"- val predictions {val_metrics['prediction_counts']}")
    (args.run / "diagnosis.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
