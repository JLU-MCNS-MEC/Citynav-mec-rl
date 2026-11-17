#!/usr/bin/env python3
"""Stage 14-R1: formal from-scratch training of the five R2 preset MEC experts.

Fixed protocol
* data    : 11B four-action teacher pool (18,521 eps / 1,279,882 states);
            episode-block shuffle, 5% blank-episode synthesis per epoch;
* dev     : deterministic 2% episode hold-out from the TRAIN pool (approved
            before training; disjoint from the 2,498 Val-Seen test tasks).
            Dev Macro-F1 selects best.pt and drives early stopping;
* loop    : seed 42, AdamW, lr 3e-4, wd 0.01, BF16 autocast, gradient clip 1.0,
            OneCycleLR per optimizer update (pct_start 0.03, cosine);
* schedule: <=20 epochs, patience 10; no Val-Seen used for selection;
* batch   : 128 for small/medium/ts/tm; 64 x grad-accum 2 for large.

Outputs
  checkpoints/<Model>/{best.pt,last.pt}
  results/<Model>/{training_history.json,training_history.jsonl,best_epoch.json,
                   config.json,train_progress.json,gpu_util.jsonl,
                   train_loss.png,dev_macro_f1.png}
  status.json (live monitoring; aggregated across the queue)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
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

KEYS = ["small", "medium", "ts", "tm", "large"]
CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]
DATA = REPO / "experiments/11_teacher_geometry_ablation" \
    "/11B_transformer_m_v4/results/data/train_seen_memmap"
LR = 3e-4
WD = 0.01
BATCH_CFG = {"small": (128, 1), "medium": (128, 1), "ts": (128, 1),
             "tm": (128, 1), "large": (64, 2)}


def load_ep_ranges() -> dict[int, tuple[int, int]]:
    ep = json.loads((DATA / "ep_ranges.json").read_text(encoding="utf-8"))
    ep = {int(k): (int(v[0]), int(v[1])) for k, v in ep.items()}
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    assert sum(e - s for s, e in ep.values()) == int(meta["n_states"])
    return ep


def make_dev_split(ep: dict[int, tuple[int, int]], seed: int,
                   dev_frac: float) -> list[int]:
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
    return metrics(np.asarray(preds), np.asarray(labels),
                   float(np.sum(losses) / max(len(labels), 1)))


def git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def write_paths(model_key: str, root: Path | None = None) -> tuple[Path, Path]:
    name = MODEL_NAMES[model_key]
    base = root if root is not None else HERE
    return base / "checkpoints" / name, base / "results" / name


def plot_curves(hist: list[dict], res_dir: Path, name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"WARN matplotlib unavailable: {e}", flush=True)
        return
    ep = [h["epoch"] for h in hist]
    plt.figure(figsize=(6, 4))
    plt.plot(ep, [h["train_loss"] for h in hist], marker="o", label="train loss")
    plt.plot(ep, [h["dev"]["loss"] for h in hist], marker="s", label="dev loss")
    plt.xlabel("epoch"); plt.ylabel("cross-entropy"); plt.grid(alpha=.3)
    plt.legend(); plt.tight_layout(); plt.savefig(res_dir / "train_loss.png", dpi=120)
    plt.close()
    plt.figure(figsize=(6, 4))
    plt.plot(ep, [h["dev"]["macro_f1"] for h in hist], marker="o",
             label="dev Macro-F1")
    best = int(np.argmax([h["dev"]["macro_f1"] for h in hist]))
    plt.scatter([ep[best]], [hist[best]["dev"]["macro_f1"]], color="red",
                zorder=5, label=f"best @ep{ep[best]}")
    plt.xlabel("epoch"); plt.ylabel("macro-F1"); plt.grid(alpha=.3)
    plt.legend(); plt.tight_layout()
    plt.savefig(res_dir / "dev_macro_f1.png", dpi=120)
    plt.close()
    print(f"[{name}] wrote train_loss.png / dev_macro_f1.png", flush=True)


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
    ap.add_argument("--out-root", default=str(HERE),
                    help="engineering test root; formal runs use the default")
    # engineering-only knobs (0 = full run); not used for formal training
    ap.add_argument("--max-train-steps", type=int, default=0)
    ap.add_argument("--dev-limit", type=int, default=0)
    args = ap.parse_args()
    if not os.environ.get("CITYNAV_DATA"):
        raise SystemExit("CITYNAV_DATA is not set")

    key = args.model
    name = MODEL_NAMES[key]
    root = Path(args.out_root)
    batch, accum = BATCH_CFG[key]
    ckpt_dir, res_dir = write_paths(key, root)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = ckpt_dir / "best.pt", ckpt_dir / "last.pt"
    if not args.resume and best_path.exists():
        raise SystemExit(f"{best_path} 已存在，拒绝覆盖（--resume 续训或换目录）")

    set_seed(args.seed)
    device = args.device
    ep = load_ep_ranges()
    dev_eps = make_dev_split(ep, args.seed, args.dev_frac)
    dev_set = set(dev_eps)
    train_eps = [k for k in sorted(ep) if k not in dev_set]
    dev_idx = split_states(ep, dev_eps)
    if args.dev_limit:
        dev_idx = dev_idx[: args.dev_limit]
    n_train = int(sum(ep[k][1] - ep[k][0] for k in train_eps))
    steps_per_epoch = math.ceil(n_train / batch)
    updates_per_epoch = math.ceil(n_train / (batch * accum))

    split_dir = root / "dev_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"seed{args.seed}_dev{args.dev_frac:.3f}.json"
    split_rec = {"seed": args.seed, "dev_frac": args.dev_frac,
                 "n_episodes_train_pool": len(ep),
                 "dev_episodes": dev_eps, "n_dev_episodes": len(dev_eps),
                 "n_dev_states": len(dev_idx),
                 "n_train_episodes": len(train_eps), "n_train_states": n_train,
                 "note": "independent Dev = 2% train-pool episode hold-out; "
                         "disjoint from the 2,498 Val-Seen closed-loop tasks"}
    if split_path.exists():
        old = json.loads(split_path.read_text(encoding="utf-8"))
        if old["dev_episodes"] != dev_eps:
            raise SystemExit(f"dev split mismatch: {split_path}")
    else:
        split_path.write_text(json.dumps(split_rec, indent=2) + "\n",
                              encoding="utf-8")

    config = vars(args).copy()
    config.update({"model_name": name, "batch": batch, "accum": accum,
                   "effective_batch": batch * accum,
                   "steps_per_epoch": steps_per_epoch,
                   "updates_per_epoch": updates_per_epoch,
                   "n_train_states": n_train, "n_dev_states": len(dev_idx),
                   "n_dev_episodes": len(dev_eps),
                   "dev_split_file": str(split_path),
                   "precision": "bf16-autocast",
                   "scheduler": "OneCycleLR per optimizer update, "
                                "total_steps=epochs*updates_per_epoch, "
                                "pct_start=0.03, cosine",
                   "loss": "cross_entropy (4-class, no class weights)",
                   "label_smoothing": 0.0,
                   "grad_clip": 1.0,
                   "best_selection": "independent Dev Macro-F1 (2% train-pool "
                                     "episode hold-out)",
                   "git_sha": git_sha(),
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    (res_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n",
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

    start_epoch, best, global_step, n_best_epochs = 0, -1.0, 0, 0
    hist: list[dict] = []
    if args.resume:
        if not last_path.exists():
            raise SystemExit(f"--resume 但 {last_path} 不存在")
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        best = float(ck.get("best", -1.0))
        global_step = int(ck.get("global_step", 0))
        start_epoch = int(ck["epoch"]) + 1
        restore_rng(ck["rng"])
        hpath = res_dir / "training_history.jsonl"
        if hpath.exists():
            for line in hpath.read_text(encoding="utf-8").splitlines():
                try:
                    hist.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            f1s = [h["dev"]["macro_f1"] for h in hist]
            if f1s:
                n_best_epochs = len(f1s) - 1 - int(np.argmax(f1s))
        print(f"resumed epoch {start_epoch} best={best:.4f} "
              f"n_best_epochs={n_best_epochs}", flush=True)

    print(f"[{name}] params={n_params:,} batch={batch}x{accum} "
          f"train_states={n_train:,} dev_states={len(dev_idx):,} "
          f"steps/epoch={steps_per_epoch:,} updates/epoch={updates_per_epoch:,} "
          f"epochs={args.epochs} patience={args.patience}", flush=True)
    status_path = root / "status.json"
    torch.cuda.reset_peak_memory_stats()

    nan_updates = 0
    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        rng = np.random.default_rng(args.seed * 1000 + epoch)
        ep_order = rng.permutation(np.asarray(train_eps, dtype=np.int64))
        idx = np.concatenate([np.arange(ep[k][0], ep[k][1], dtype=np.int64)
                              for k in ep_order])
        blank = build_blank_mask(args.seed, epoch, ep_order, ep, args.blank_frac)
        ds = ChunkedMemmapDataset("train_seen", idx, blank_mask=blank)
        loader = DataLoader(ds, batch_size=batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate11,
                            persistent_workers=args.workers > 0,
                            prefetch_factor=4 if args.workers else None)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, count, correct, gnorm_sum, updates, micro = 0.0, 0, 0, 0.0, 0, 0
        window_ok = True
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
            if finite:
                total += float(loss.detach()) * len(label)
                correct += int((logits.detach().argmax(-1) == label).sum())
            count += len(label)
            if micro % accum == 0:
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                if window_ok and math.isfinite(gn):
                    optimizer.step()
                    scheduler.step()
                    updates += 1
                    global_step += 1
                    gnorm_sum += gn
                elif window_ok:
                    nan_updates += 1
                optimizer.zero_grad(set_to_none=True)
                window_ok = True
            if updates and updates % 1000 == 0 and micro % accum == 0:
                el = time.time() - started
                print(f"  epoch {epoch} update {updates}/{updates_per_epoch} "
                      f"loss={total / max(count, 1):.4f} "
                      f"acc={correct / max(count, 1):.4f} {el:.0f}s", flush=True)
            if args.max_train_steps and micro >= args.max_train_steps:
                break
        train_loss = total / max(count, 1)
        train_acc = correct / max(count, 1)
        dev = evaluate_dev(model, encoder, dev_idx, device)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        rec = {"epoch": epoch, "train_loss": train_loss,
               "train_acc": train_acc, "train_micro_steps": micro,
               "optimizer_steps": updates, "global_step": global_step,
               "grad_norm": gnorm_sum / max(updates, 1), "dev": dev,
               "lr": scheduler.get_last_lr()[0],
               "seconds": round(time.time() - started, 1),
               "peak_mem_gb": peak_gb, "gpu": gpu_snapshot(),
               "nan_updates": nan_updates}
        hist.append(rec)
        with (res_dir / "training_history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        (res_dir / "training_history.json").write_text(
            json.dumps(hist, indent=1) + "\n", encoding="utf-8")
        with (res_dir / "gpu_util.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"epoch": epoch, "peak_mem_gb": peak_gb,
                                 **rec["gpu"]}) + "\n")

        improved = dev["macro_f1"] > best
        if improved:
            best = float(dev["macro_f1"])
            n_best_epochs = 0
            torch.save({"model": model.state_dict(), "config": config,
                        "epoch": epoch, "dev": dev, "macro_f1": best},
                       best_path)
        else:
            n_best_epochs += 1
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "epoch_done": True, "config": config, "best": best,
                    "global_step": global_step, "rng": rng_state()}, last_path)
        best_epoch_rec = {
            "best_epoch": int(np.argmax([h["dev"]["macro_f1"] for h in hist])),
            "best_dev_macro_f1": best,
            "selected_epoch_dev_f1": dev["macro_f1"],
            "improved": improved, "epoch": epoch,
            "checkpoint": str(best_path), "updated":
                time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        (res_dir / "best_epoch.json").write_text(
            json.dumps(best_epoch_rec, indent=2) + "\n", encoding="utf-8")
        (res_dir / "train_progress.json").write_text(json.dumps({
            "model": name, "epoch": epoch, "epochs": args.epochs,
            "optimizer_step": global_step, "micro_step": micro,
            "train_loss": train_loss, "dev_macro_f1": dev["macro_f1"],
            "best": best, "lr": float(scheduler.get_last_lr()[0]),
            "grad_norm": rec["grad_norm"], "gpu": rec["gpu"],
            "epoch_seconds": rec["seconds"], "peak_mem_gb": peak_gb,
            "patience_left": args.patience - n_best_epochs,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
            indent=2) + "\n", encoding="utf-8")
        status_path.write_text(json.dumps({
            "stage": "train", "model": name, "key": key, "epoch": epoch,
            "epochs": args.epochs, "optimizer_step": global_step,
            "micro_step": micro, "train_loss": train_loss,
            "dev_macro_f1": dev["macro_f1"], "best": best,
            "lr": float(scheduler.get_last_lr()[0]), "grad_norm": rec["grad_norm"],
            "gpu": rec["gpu"], "peak_mem_gb": peak_gb,
            "epoch_seconds": rec["seconds"],
            "patience_left": args.patience - n_best_epochs,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
            indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "epoch": epoch, "loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "dev_loss": round(dev["loss"], 4),
            "dev_acc": round(dev["accuracy"], 4),
            "dev_macro_f1": round(dev["macro_f1"], 4),
            "per_class_f1": {c: round(dev["per_class"][c]["f1"], 4)
                             for c in CLASSES},
            "lr": round(float(scheduler.get_last_lr()[0]), 7),
            "grad_norm": round(rec["grad_norm"], 4), "best": round(best, 4),
            "patience_left": args.patience - n_best_epochs,
            "peak_mem_gb": round(peak_gb, 2), "nan_updates": nan_updates,
            "gpu": rec["gpu"], "sec": rec["seconds"]}), flush=True)
        if n_best_epochs >= args.patience:
            print(f"early stop: no dev macro_f1 improvement for {args.patience} "
                  f"epochs (best {best:.4f})", flush=True)
            break
    plot_curves(hist, res_dir, name)
    (res_dir / "train_progress.json").write_text(json.dumps({
        "model": name, "epoch": hist[-1]["epoch"] if hist else -1,
        "epochs": args.epochs, "done": True, "best": best,
        "best_epoch": int(np.argmax([h["dev"]["macro_f1"] for h in hist]))
        if hist else -1,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2) + "\n",
        encoding="utf-8")
    status_path.write_text(json.dumps({
        "stage": "train_done", "model": name, "key": key,
        "best": best, "epochs_run": len(hist),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
