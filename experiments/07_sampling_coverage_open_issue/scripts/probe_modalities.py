#!/usr/bin/env python3
"""Inference-only input probe on the frozen Stage-05 checkpoint.

This is NOT a retrained ablation: the model is untouched and the probe feeds it
perturbed inputs (zeroed belief / zeroed history / zeroed target embedding) on
teacher-forced validation states.  Zeroing an input is far outside the training
distribution, so the accuracy drop is a *hint*, not a causal attribution.  It is
recorded to motivate the real ablation matrix, not to replace it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "05_transformer_m_training/scripts"))
sys.path.insert(0, str(HERE))

from dataset_v2 import CACHE, EpisodeStore, FrozenEncoder  # noqa: E402
from model_v2 import TransformerM  # noqa: E402
from train_v2 import SampleDataset, collate, metrics, to_model_batch  # noqa: E402

CKPT = ROOT / "05_transformer_m_training/runs_v2/seed42/best.pt"
OUT = HERE.parent / "results"


def evaluate(model, encoder, store, pairs, device, perturb: str = "none") -> dict:
    loader = DataLoader(SampleDataset(store, pairs), batch_size=64, shuffle=False,
                        num_workers=4, collate_fn=collate)
    preds, labels = [], []
    for batch in loader:
        if perturb == "belief":
            batch["belief"] = np.zeros_like(batch["belief"])
        elif perturb == "history":
            batch["history_mask"] = np.zeros_like(batch["history_mask"])
            batch["history_vis"] = np.zeros_like(batch["history_vis"])
        elif perturb == "target":
            batch["target_emb"] = np.zeros_like(batch["target_emb"])
        elif perturb == "current":
            batch["image"] = np.zeros_like(batch["image"])
        with torch.inference_mode():
            logits = model(to_model_batch(batch, encoder, device))
        preds.extend(logits.argmax(-1).cpu().numpy().tolist())
        labels.extend(batch["label"].tolist())
    return metrics(np.asarray(preds), np.asarray(labels))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--per-episode", type=int, default=4)
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    store = EpisodeStore(args.split, CACHE)
    rng = np.random.default_rng(7)
    ids = np.arange(len(store))[:args.episodes]
    pairs = []
    for e in ids:
        T = store.length(int(e))
        for t in rng.choice(T, size=min(args.per_episode, T), replace=False):
            pairs.append((int(e), int(t)))
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = TransformerM().to(args.device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    encoder = FrozenEncoder(args.device)

    out = {"checkpoint": str(CKPT), "split": args.split, "states": len(pairs),
           "note": "inference-only perturbation; not a retrained ablation",
           "conditions": {}}
    for cond in ("none", "belief", "history", "target", "current"):
        m = evaluate(model, encoder, store, pairs, args.device, cond)
        out["conditions"][cond] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                                   "stop_recall": m["per_class"]["Stop"]["recall"]}
        print(cond, {k: round(v, 4) for k, v in out["conditions"][cond].items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "modality_probe.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'modality_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
