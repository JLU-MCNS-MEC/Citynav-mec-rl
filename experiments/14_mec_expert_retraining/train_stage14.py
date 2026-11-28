#!/usr/bin/env python3
"""Stage 14: formal from-scratch training of the five R2 preset MEC models.

Protocol (fixed for this stage):
* data     : 11B pure four-action teacher pool (18,521 eps / 1,279,882 states);
             episode-block shuffle per epoch, 5% blank-episode synthesis;
* dev      : deterministic 2% episode hold-out from the TRAIN pool (independent
             of the 2,498 Val-Seen closed-loop tasks); Dev Macro-F1 selects
             best.pt and drives early stopping (patience 10);
* loop     : seed 42, AdamW, lr 3e-4, wd 0.01, BF16 autocast, OneCycleLR per
             optimizer update, <=20 epochs;
* batch    : 128 for small/medium/ts/tm; 64 x grad-accum 2 for large;
* outputs  : runs/<key>/seed42/{best.pt, epoch_NN.pt, checkpoint.pt,
             history.jsonl, config.json, train_progress.json, gpu_util.jsonl}.

No history checkpoint is loaded unless --resume; the R2 preset architectures in
models.py are used unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
assert (REPO / "experiments").is_dir(), f"REPO 解析错误: {REPO}"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments/11_teacher_geometry_ablation"
                       "/11B_transformer_m_v4/scripts"))
sys.path.insert(0, str(REPO / "experiments/10_goal_query_mec_training"
                       "/10D_transformer_m_training/scripts"))
sys.path.insert(0, str(REPO / "experiments/05_transformer_m_training/scripts"))
sys.path.insert(0, str(REPO / "experiments/04_visual_goal_global_map_pipeline/scripts"))

from models import MODEL_NAMES, build_model  # noqa: E402
from dataset11b import ChunkedMemmapDataset  # noqa: E402
from dataset_v2 import FrozenEncoder  # noqa: E402
from train_10d import gpu_snapshot, metrics, restore_rng, rng_state, set_seed  # noqa: E402
from train_11b import build_blank_mask, collate11, to_model_batch11  # noqa: E402
from v5fix_contract import BLANK_FRAC  # noqa: E402

KEYS = ["small", "small1g", "medium", "ts", "tm", "large"]
CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]
DATA = HERE.parents[1] / "experiments/11_teacher_geometry_ablation" \
    "/11B_transformer_m_v4/results/data/train_seen_memmap"
LR = 3e-4
WD = 0.01
BATCH_CFG = {"small": (128, 1), "small1g": (128, 1), "medium": (128, 1), "ts": (128, 1),
             "tm": (128, 1), "large": (64, 2)}


def load_ep_ranges() -> tuple[dict[int, tuple[int, int]], int]:
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    n_states = int(meta["n_states"])
    ep = json.loads((DATA / "ep_ranges.json").read_text(encoding="utf-8"))
    ep = {int(k): (int(v[0]), int(v[1])) for k, v in ep.items()}
    assert sum(e - s for s, e in ep.values()) == n_states, "ep_ranges mismatch"
    return ep, n_states


def make_dev_split(ep: dict[int, tuple[int, int]], seed: int,
                   dev_frac: float) -> list[int]:
    """Deterministic episode-level hold-out, independent of Val-Seen test tasks."""
    ep_list = np.asarray(sorted(ep), dtype=np.int64)
    rng = np.random.default_rng(seed * 7777 + 14)
    perm = rng.permutation(ep_list)
    n_dev = max(1, int(round(len(ep_list) * dev_frac)))
    return sorted(int(x) for x in perm[:n_dev])


def split_states(ep: dict[int, tuple[int, int]], episodes: list[int]) -> np.ndarray:
    if not episodes:
        return np.zeros(0, np.int64)
    return np.concatenate([np.arange(ep[k][0], ep[k][1], dtype=np.int64)
                           for k in episodes])


@torch.no_grad()
def evaluate_dev(model, encoder, dev_idx: np.ndarray, device: str,
                 batch_size: int = 128) -> dict:
    """Dev Macro-F1 under BF16 autocast (same numerics as training)."""
    model.eval()
    ds = ChunkedMemmapDataset("train_seen", dev_idx, blank_mask=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate11)
    preds, labels, losses = [], [], []
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(to_model_batch11(batch, encoder, device))
            loss = F.cross_entropy(logits, torch.from_numpy(batch["label"]).to(device))
        losses.append(float(loss.item()) * len(batch["label"]))
        preds.extend(logits.argmax(-1).cpu().numpy().tolist())
        labels.extend(batch["label"].tolist())
    mean_loss = float(np.sum(losses) / max(len(labels), 1))
    return metrics(np.asarray(preds), np.asarray(labels), mean_loss)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=KEYS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--weight-decay", type=float, default=WD)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--blank-frac", type=float, default=BLANK_FRAC)
    ap.add_argument("--dev-frac", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-root", default=str(HERE / "runs"))
    # test-only knobs (0 = full run); do not use for the formal runs
    ap.add_argument("--max-train-steps", type=int, default=0)
    ap.add_argument("--dev-limit", type=int, default=0)
    args = ap.parse_args()
    if not os.environ.get("CITYNAV_DATA"):
        raise SystemExit("CITYNAV_DATA is not set")

    key = args.model
    name = MODEL_NAMES[key]
    batch, accum = BATCH_CFG[key]
    out_root = Path(args.out_root)
    run_dir = out_root / key / f"seed{args.seed}"
    if not args.resume and (run_dir / "best.pt").exists():
        raise SystemExit(f"{run_dir} 已存在 best.pt，拒绝覆盖（用 --resume 续训）")
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = args.device
    ep, n_states = load_ep_ranges()
    dev_eps = make_dev_split(ep, args.seed, args.dev_frac)
    train_eps = [k for k in sorted(ep) if k not in set(dev_eps)]
    dev_idx = split_states(ep, dev_eps)
    if args.dev_limit:
        dev_idx = dev_idx[: args.dev_limit]
    n_train = int(sum(ep[k][1] - ep[k][0] for k in train_eps))
    n_dev = len(dev_idx)
    steps_per_epoch = math.ceil(n_train / batch)
    updates_per_epoch = math.ceil(n_train / (batch * accum))

    split_path = out_root / "split" / f"seed{args.seed}_dev{args.dev_frac:.3f}.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_rec = {"seed": args.seed, "dev_frac": args.dev_frac,
                 "n_episodes_train_pool": len(ep),
                 "dev_episodes": dev_eps, "n_dev_episodes": len(dev_eps),
                 "n_dev_states": len(dev_idx),
                 "n_train_episodes": len(train_eps), "n_train_states": n_train,
                 "note": "dev hold-out from the training pool; disjoint from "
                         "the 2,498 Val-Seen closed-loop tasks"}
    if split_path.exists():
        old = json.loads(split_path.read_text(encoding="utf-8"))
        if old["dev_episodes"] != dev_eps:
            raise SystemExit(f"dev split mismatch: {split_path}")
    else:
        split_path.write_text(json.dumps(split_rec, indent=2) + "\n",
                              encoding="utf-8")

    config = vars(args).copy()
    config.update({"model_name": name, "batch": batch, "accum": accum,
                   "steps_per_epoch": steps_per_epoch,
                   "updates_per_epoch": updates_per_epoch,
                   "n_train_states": n_train, "n_dev_states": n_dev,
                   "n_dev_episodes": len(dev_eps),
                   "precision": "bf16-autocast",
                   "scheduler": "OneCycleLR per optimizer update, "
                                "total_steps=epochs*updates_per_epoch",
                   "best_selection": "independent Dev Macro-F1 (2% train-pool "
                                     "episode hold-out)"})
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n",
                                         encoding="utf-8")

    model = build_model(key).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_updates = args.epochs * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_updates,
        pct_start=args.warmup, anneal_strategy="cos", cycle_momentum=False)
    encoder = FrozenEncoder(device)

    start_epoch, best, global_step = 0, -1.0, 0
    n_best_epochs = 0
    ckpt_path = run_dir / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best = float(ckpt.get("best", -1.0))
        global_step = int(ckpt.get("global_step", 0))
        start_epoch = int(ckpt["epoch"]) + 1
        restore_rng(ckpt["rng"])
        hpath = run_dir / "history.jsonl"
        if hpath.exists():
            f1s = []
            for line in hpath.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    f1s.append(float(rec["dev"]["macro_f1"]))
                except (json.JSONDecodeError, KeyError):
                    continue
            if f1s:
                n_best_epochs = len(f1s) - 1 - int(np.argmax(f1s))
        print(f"resumed at epoch {start_epoch} (best={best:.4f} "
              f"n_best_epochs={n_best_epochs})", flush=True)

    print(f"[{name}] params={n_params:,} batch={batch}x{accum} "
          f"train_states={n_train:,} dev_states={n_dev:,} "
          f"steps/epoch={steps_per_epoch:,} updates/epoch={updates_per_epoch:,} "
          f"epochs={args.epochs} patience={args.patience} run={run_dir}",
          flush=True)
    torch.cuda.reset_peak_memory_stats()

    nan_updates = 0
    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        rng = np.random.default_rng(args.seed * 1000 + epoch)
        ep_order = rng.permutation(np.asarray(train_eps, dtype=np.int64))
        idx = np.concatenate([np.arange(ep[k][0], ep[k][1], dtype=np.int64)
                              for k in ep_order])
        blank = build_blank_mask(args.seed, epoch, ep_order, ep,
                                 args.blank_frac)
        ds = ChunkedMemmapDataset("train_seen", idx, blank_mask=blank)
        loader = DataLoader(ds, batch_size=batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate11,
                            persistent_workers=args.workers > 0,
                            prefetch_factor=4 if args.workers else None)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, count, correct, gnorm_sum, updates, micro = 0.0, 0, 0, 0.0, 0, 0
        window_ok = True     # no non-finite loss inside the current accumulation
        for batch_data in loader:
            micro += 1
            label = torch.from_numpy(batch_data["label"]).to(device)
            mb = to_model_batch11(batch_data, encoder, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(mb)
                loss = F.cross_entropy(logits, label)
            finite = bool(torch.isfinite(loss).item())
            if finite:
                (loss / accum).backward()
            else:
                nan_updates += 1
                window_ok = False
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(label) if finite else 0.0
            correct += int((logits.detach().argmax(-1) == label).sum())
            count += len(label)
            if micro % accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                gnorm = float(gn)
                if window_ok and math.isfinite(gnorm):
                    optimizer.step()
                    scheduler.step()
                    updates += 1
                    global_step += 1
                    gnorm_sum += gnorm
                elif window_ok:
                    nan_updates += 1
                optimizer.zero_grad(set_to_none=True)
                window_ok = True
            if updates and updates % 1000 == 0 and micro % accum == 0:
                el = time.time() - started
                print(f"  epoch {epoch} update {updates}/{updates_per_epoch} "
                      f"loss={total / count:.4f} acc={correct / count:.4f} "
                      f"{el:.0f}s", flush=True)
            if args.max_train_steps and micro >= args.max_train_steps:
                break
        train_loss = total / max(count, 1)
        train_acc = correct / max(count, 1)
        dev = evaluate_dev(model, encoder, dev_idx, device)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        rec = {"epoch": epoch, "train_loss": train_loss,
               "train_acc": train_acc, "train_micro_steps": micro,
               "updates": updates, "grad_norm": gnorm_sum / max(updates, 1),
               "dev": dev, "lr": scheduler.get_last_lr()[0],
               "seconds": round(time.time() - started, 1),
               "global_step": global_step, "peak_mem_gb": peak_gb,
               "gpu": gpu_snapshot(), "nan_updates": nan_updates}
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        with (run_dir / "gpu_util.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"epoch": epoch, "peak_mem_gb": peak_gb,
                                 **rec["gpu"]}) + "\n")
        torch.save({"model": model.state_dict(), "config": config,
                    "epoch": epoch, "dev": dev,
                    "train": {"loss": train_loss, "acc": train_acc},
                    "best": max(best, dev["macro_f1"])},
                   run_dir / f"epoch_{epoch:02d}.pt")
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "step": 0, "epoch_done": True, "config": config,
                    "best": best, "global_step": global_step,
                    "rng": rng_state()}, ckpt_path)
        improved = dev["macro_f1"] > best
        if improved:
            best = float(dev["macro_f1"])
            n_best_epochs = 0
            torch.save({"model": model.state_dict(), "config": config,
                        "epoch": epoch, "dev": dev, "macro_f1": best},
                       run_dir / "best.pt")
            print(f"  new best dev macro_f1={best:.4f} at epoch {epoch}",
                  flush=True)
        else:
            n_best_epochs += 1
        print(json.dumps({
            "epoch": epoch, "loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "dev_loss": round(dev["loss"], 4),
            "dev_acc": round(dev["accuracy"], 4),
            "dev_macro_f1": round(dev["macro_f1"], 4),
            "per_class_f1": {c: round(dev["per_class"][c]["f1"], 4)
                             for c in CLASSES},
            "lr": round(scheduler.get_last_lr()[0], 7),
            "grad_norm": round(rec["grad_norm"], 4), "best": round(best, 4),
            "patience_left": args.patience - n_best_epochs,
            "peak_mem_gb": round(peak_gb, 2), "nan_updates": nan_updates,
            "gpu": rec["gpu"], "sec": rec["seconds"]}), flush=True)
        with (run_dir / "train_progress.json").open("w", encoding="utf-8") as fh:
            json.dump({"epoch": epoch, "epochs": args.epochs,
                       "global_step": global_step, "train_loss": train_loss,
                       "dev_macro_f1": dev["macro_f1"], "best": best,
                       "patience_left": args.patience - n_best_epochs,
                       "seconds": rec["seconds"],
                       "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                      fh, indent=2)
        if n_best_epochs >= args.patience:
            print(f"early stop: no dev macro_f1 improvement for {args.patience} "
                  f"epochs (best {best:.4f})", flush=True)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
