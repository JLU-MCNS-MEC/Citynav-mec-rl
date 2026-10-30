#!/usr/bin/env python3
"""Per-class diagnostics for a trained stage-05 model.

Evaluates teacher-forced states (history from the teacher trajectory, exactly as
in training) on Train-Seen / Val-Seen / Val-Unseen.  Reports accuracy, macro-F1,
per-class precision/recall/F1, confusion matrices and the predicted action mix,
plus STOP behaviour by remaining distance to the target (a diagnostic only - the
model never sees that distance).

Outputs: results/diagnostics_{tag}.json, figures/confusion_{tag}.png,
figures/per_class_{tag}.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset_v2 import CACHE, EpisodeStore, load_split  # noqa: E402
from model_v2 import TransformerM  # noqa: E402
from train_v2 import (CLASSES, FrozenEncoder, evaluate, fixed_pairs,  # noqa: E402
                      metrics, SampleDataset, collate, to_model_batch)

FIG = HERE.parent / "figures"
RESULTS = HERE.parent / "results"
SPLITS = ("train_seen", "val_seen", "val_unseen")


def stop_by_distance(cache: EpisodeStore, pairs, preds, teacher: dict) -> dict:
    """STOP prediction rate vs the true remaining distance (diagnostic only)."""
    buckets = [(0, 20), (20, 30), (30, 50), (50, 100), (100, 200), (200, 10 ** 9)]
    labels = ["<=20", "20-30", "30-50", "50-100", "100-200", ">200"]
    counts = np.zeros(len(buckets))
    stops = np.zeros(len(buckets))
    for (e, t), p in zip(pairs, preds):
        key = cache.episodes[e]["key"]
        target = teacher[key].target
        states = cache.array(e)["states"]
        d = float(np.linalg.norm(states[t, :2] - target))
        for i, (lo, hi) in enumerate(buckets):
            if d <= hi and (i == 0 or d > lo):
                counts[i] += 1
                stops[i] += int(p == 0)
                break
    return {name: {"decisions": int(counts[i]), "stop_rate": float(stops[i] / counts[i])
                   if counts[i] else 0.0} for i, name in enumerate(labels)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="seed42")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--per-episode", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_dir = HERE.parent / "runs_v2" / args.run
    ckpt = torch.load(run_dir / args.checkpoint, map_location=args.device, weights_only=False)
    model = TransformerM().to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    encoder = FrozenEncoder(args.device)
    tag = run_dir.name
    report = {"run": tag, "checkpoint_epoch": ckpt.get("epoch"), "splits": {}}
    FIG.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    for split in SPLITS:
        cache = EpisodeStore(split, CACHE)
        teacher = {ep.key: ep for ep in load_split(split)}
        ids = np.arange(len(cache))[:args.episodes]
        pairs = fixed_pairs(cache, args.per_episode, seed=11, episodes=ids)
        out = evaluate(model, encoder, cache, pairs, args.device, batch_size=args.batch)
        # reuse the evaluated predictions for the distance diagnostic
        dataset = SampleDataset(cache, pairs)
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=4,
                            collate_fn=collate)
        preds = []
        with torch.no_grad():
            for batch in loader:
                logits = model(to_model_batch(batch, encoder, args.device))
                preds.extend(logits.argmax(-1).cpu().numpy().tolist())
        out["stop_by_distance"] = stop_by_distance(cache, pairs, preds, teacher)
        report["splits"][split] = out
        print(f"{split}: n={out['n']} acc={out['accuracy']:.4f} macro_f1={out['macro_f1']:.4f} "
              f"stop_recall={out['per_class']['Stop']['recall']:.3f}", flush=True)
    (RESULTS / f"diagnostics_{tag}.json").write_text(json.dumps(report, indent=2) + "\n",
                                                     encoding="utf-8")
    plot(report, tag)
    return 0


def plot(report: dict, tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = list(report["splits"])
    fig, axes = plt.subplots(1, len(splits), figsize=(4 * len(splits), 3.6))
    for ax, split in zip(np.atleast_1d(axes), splits):
        conf = np.asarray(report["splits"][split]["confusion"])
        norm = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(4))
        ax.set_xticklabels(CLASSES, rotation=35, fontsize=7)
        ax.set_yticks(range(4))
        ax.set_yticklabels(CLASSES, fontsize=7)
        ax.set_xlabel("predicted")
        ax.set_ylabel("label")
        ax.set_title(f"{split} (row-normalised)", fontsize=8)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG / f"confusion_{tag}.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    width = 0.8 / len(splits)
    for i, split in enumerate(splits):
        pc = report["splits"][split]["per_class"]
        x = np.arange(4) + i * width
        axes[0].bar(x, [pc[c]["recall"] for c in CLASSES], width=width, label=split)
        axes[1].bar(x, [pc[c]["f1"] for c in CLASSES], width=width, label=split)
    for ax, title in zip(axes, ("recall", "F1")):
        ax.set_xticks(np.arange(4) + width * (len(splits) - 1) / 2)
        ax.set_xticklabels(CLASSES, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / f"per_class_{tag}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
