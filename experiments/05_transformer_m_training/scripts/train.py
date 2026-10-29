#!/usr/bin/env python3
"""Small-scale Transformer-M training / overfit diagnosis (seed 42, 43)."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

from data import EpisodeStore, collate, episode_batches  # noqa: E402
from model import N_ACTIONS, TransformerM  # noqa: E402

CANDIDATES = HERE.parent / "04_visual_goal_global_map_pipeline/cache/candidates"
CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(pred: np.ndarray, label: np.ndarray) -> dict:
    out = {"accuracy": float((pred == label).mean()) if len(label) else 0.0,
           "n": int(len(label))}
    per_class = {}
    for c, name in enumerate(CLASSES):
        tp = int(((pred == c) & (label == c)).sum())
        fp = int(((pred == c) & (label != c)).sum())
        fn = int(((pred != c) & (label == c)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1,
                           "support": int((label == c).sum())}
    out["per_class"] = per_class
    out["macro_f1"] = float(np.mean([v["f1"] for v in per_class.values()]))
    conf = np.zeros((N_ACTIONS, N_ACTIONS), int)
    for p, l in zip(pred, label):
        conf[l, p] += 1
    out["confusion"] = conf.tolist()
    out["prediction_counts"] = {name: int((pred == c).sum()) for c, name in enumerate(CLASSES)}
    return out


def evaluate(model, store, indices, device, max_samples: int = 4000, seed: int = 0) -> dict:
    model.eval()
    preds, labels = [], []
    total = 0
    with torch.no_grad():
        for batch_ids in episode_batches(indices, store, 64, shuffle=True, seed=seed):
            samples = [store.sample(i, t) for i, t in batch_ids]
            batch = collate(samples)
            logits = model({k: v.to(device) for k, v in batch.items() if k != "label"})
            preds.extend(logits.argmax(-1).cpu().numpy().tolist())
            labels.extend(batch["label"].numpy().tolist())
            total += len(samples)
            if total >= max_samples:
                break
    return metrics(np.asarray(preds), np.asarray(labels))


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, config: dict) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "config": config,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, model, optimizer, scheduler) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    if torch.cuda.is_available() and "cuda_rng" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return int(payload["epoch"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=HERE / "features")
    parser.add_argument("--selection", type=Path, default=HERE / "results/selection.json")
    parser.add_argument("--out", type=Path, default=HERE / "runs/seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    store = EpisodeStore(args.features, CANDIDATES)
    model = TransformerM(args.d_model, args.layers, args.heads).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    config = vars(args) | {"n_params": model.n_params()}
    (args.out / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"model params: {model.n_params():,}  device={args.device}", flush=True)
    print(f"train={len(selection['train'])} val={len(selection['val'])}", flush=True)

    start_epoch = 0
    ckpt = args.out / "checkpoint.pt"
    if args.resume and ckpt.exists():
        start_epoch = load_checkpoint(ckpt, model, optimizer, scheduler)
        print(f"resumed from epoch {start_epoch}", flush=True)

    history = []
    best_acc = -1.0
    hist_path = args.out / "history.json"
    if hist_path.exists():
        history = json.loads(hist_path.read_text(encoding="utf-8"))
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        start = time.time()
        total_loss, total_n, correct = 0.0, 0, 0
        grad_norms = []
        for step, batch_ids in enumerate(episode_batches(
                selection["train"], store, args.batch_size, shuffle=True, seed=args.seed + epoch)):
            samples = [store.sample(i, t) for i, t in batch_ids]
            batch = collate(samples)
            labels = batch.pop("label").to(args.device)
            batch = {k: v.to(args.device) for k, v in batch.items()}
            logits = model(batch)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norms.append(float(gn))
            optimizer.step()
            total_loss += float(loss) * len(labels)
            total_n += len(labels)
            correct += int((logits.argmax(-1) == labels).sum())
            if args.max_batches and step + 1 >= args.max_batches:
                break
        scheduler.step()
        train_loss = total_loss / max(total_n, 1)
        train_acc = correct / max(total_n, 1)
        val = evaluate(model, store, selection["val"], args.device, max_samples=4000, seed=epoch)
        train_metric = evaluate(model, store, selection["train"], args.device,
                                max_samples=2000, seed=epoch + 1)
        entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_batch_acc": train_acc,
            "train_eval": train_metric,
            "val": val,
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - start, 1),
        }
        history.append(entry)
        hist_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        save_checkpoint(ckpt, model, optimizer, scheduler, epoch + 1, config)
        if val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            save_checkpoint(args.out / "best.pt", model, optimizer, scheduler, epoch + 1, config)
        print(f"epoch {epoch+1}/{args.epochs} loss={train_loss:.4f} acc={train_acc:.3f} "
              f"val_acc={val['accuracy']:.3f} val_stop_r={val['per_class']['Stop']['recall']:.3f} "
              f"val_turn_r={val['per_class']['TurnLeft']['recall']:.3f}/"
              f"{val['per_class']['TurnRight']['recall']:.3f} {entry['seconds']}s", flush=True)

    best = max(history, key=lambda e: e["val"]["accuracy"]) if history else None
    (args.out / "final_metrics.json").write_text(
        json.dumps({"best": best, "last": history[-1] if history else None}, indent=2) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
