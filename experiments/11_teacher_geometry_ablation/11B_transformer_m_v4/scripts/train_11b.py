#!/usr/bin/env python3
"""11B: Transformer-M v4 training on the pure Stage 03 four-action teacher pool.

* data:    11B memmap (18,521 episodes / 1,279,882 decision states, uncapped,
           no resampling, no mixing -- pure teacher supervision); chunked
           memmap DataLoader, no whole-pool RAM loading (doc 4.3);
* model:   TransformerM11 (model11b.py): v4 fixes -- history time position
           encoding, deduplicated 12-d state token;
* loop:    v4 schedule fix -- per-BATCH OneCycleLR (total_steps = epochs x
           steps_per_epoch, pct_start 0.03, cosine anneal) instead of v3's
           per-epoch stepping; NaN guard skips non-finite updates; resumable
           checkpoint.pt (model/opt/sched/RNG/batch position);
* eval:    fixed val memmap (400 Val-Seen episodes x 4 states, Stage-03-style
           four-action teacher labels, VALIDATION ONLY) per epoch with loss,
           accuracy, Macro-F1, per-class P/R/F1, confusion, STOP/turn recall;
           best.pt by Val Macro-F1; early stop patience 10;
* outputs: runs_11b/seed42/{best.pt, epoch_NN.pt, checkpoint.pt, history.jsonl,
           config.json, train_progress.json, gpu_util.jsonl}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
assert (REPO / "experiments").is_dir(), f"REPO 解析错误: {REPO}"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments/10_goal_query_mec_training/10D_transformer_m_training/scripts"))
sys.path.insert(0, str(REPO / "experiments/05_transformer_m_training/scripts"))
sys.path.insert(0, str(REPO / "experiments/04_visual_goal_global_map_pipeline/scripts"))

from dataset_v2 import FrozenEncoder  # noqa: E402
from model11b import TransformerM11  # noqa: E402
from model11c import TransformerM12  # noqa: E402
from model11c_fix import TransformerM12Fix  # noqa: E402
from model11_small import SmallMEC4  # noqa: E402
from model11_medium import MediumMEC4  # noqa: E402
from dataset11b import ChunkedMemmapDataset, ValMemmapDataset  # noqa: E402
from v5fix_contract import BLANK_FRAC  # noqa: E402
from train_10d import (gpu_snapshot, metrics,  # noqa: E402
                       restore_rng, rng_state, save_checkpoint, set_seed)

CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]

MODEL_CLASSES = {"v4": TransformerM11, "v5": TransformerM12,
                 "v5fix": TransformerM12Fix,
                 "small": SmallMEC4, "medium": MediumMEC4}
MODEL_NAMES = {"v4": "TransformerM11", "v5": "TransformerM12",
               "v5fix": "TransformerM12Fix",
               "small": "SmallMEC4", "medium": "MediumMEC4"}
RUN_DIRS = {"v4": "runs_11b", "v5": "runs_11c", "v5fix": "runs_11c_v5fix",
            "small": "runs_small", "medium": "runs_medium"}


def build_blank_mask(seed: int, epoch: int, ep_order: np.ndarray,
                     ep_ranges: dict[int, tuple[int, int]],
                     blank_frac: float) -> np.ndarray:
    """V5 修正 5a：每 epoch 确定性选择 blank_frac 比例 episode 全程 blank。

    独立于 shuffle rng（seed*10000+epoch），(seed, epoch) 相同则完全复现；
    返回与 idx 拼接顺序同长的 bool 数组（True=该状态用空候选常量替换）。
    """
    rng = np.random.default_rng(seed * 10000 + epoch)
    blank_ep = rng.random(len(ep_order)) < blank_frac
    parts = [np.full(e - s, bool(blank_ep[i]), dtype=bool)
             for i, k in enumerate(ep_order) for s, e in [ep_ranges[k]]]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=bool)


def resolve_run_dir(model: str, seed: int, resume: bool,
                    results_root: Path | None = None) -> Path:
    """run 目录解析 + 旧权重保护：目标目录已有 best.pt 且非 --resume 时拒绝启动。"""
    base = results_root if results_root is not None else (HERE.parent / "results")
    run_dir = base / RUN_DIRS[model] / f"seed{seed}"
    if not resume and (run_dir / "best.pt").exists():
        raise SystemExit(
            f"run_dir {run_dir} 已存在 best.pt，拒绝覆盖旧权重"
            f"（用 --resume 续训或换目录）")
    return run_dir


def collate11(samples: list[dict]) -> dict:
    out = {
        "image": np.stack([s["image"] for s in samples]),
        "target_emb": np.stack([s["target_emb"] for s in samples]),
        "history_vis": np.stack([s["history_vis"] for s in samples]),
        "history_pose": np.stack([s["history_pose"] for s in samples]),
        "history_action": np.stack([s["history_action"] for s in samples]),
        "history_mask": np.stack([s["history_mask"] for s in samples]),
        "belief": np.stack([s["belief"] for s in samples]),
        "match": np.stack([s["match"] for s in samples]),
        "states": np.stack([s["states"] for s in samples]),
        "landmarks": np.stack([s["landmarks"] for s in samples]),
        "label": np.asarray([s["label"] for s in samples], np.int64),
    }
    return out


def to_model_batch11(batch: dict, encoder: FrozenEncoder, device: str) -> dict:
    out = {"current": encoder(batch["image"])}
    for key in ("target_emb", "history_vis", "history_pose", "history_action",
                "belief", "match", "states", "landmarks"):
        out[key] = torch.from_numpy(batch[key]).to(device)
    out["history_mask"] = torch.from_numpy(batch["history_mask"]).to(device)
    return out


@torch.no_grad()
def evaluate11(model, encoder, dataset, device, batch_size=128) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate11)
    preds, labels, losses = [], [], []
    for batch in loader:
        logits = model(to_model_batch11(batch, encoder, device))
        label = torch.from_numpy(batch["label"]).to(device)
        losses.append(float(F.cross_entropy(logits, label).item()) * len(label))
        preds.extend(logits.argmax(-1).cpu().numpy().tolist())
        labels.extend(batch["label"].tolist())
    mean_loss = float(np.sum(losses) / max(len(labels), 1))
    return metrics(np.asarray(preds), np.asarray(labels), mean_loss)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["v4", "v5", "v5fix", "small", "medium"],
                        default="v4",
                        help="v4: TransformerM11 (runs_11b); v5: TransformerM12 "
                             "(runs_11c); v5fix: TransformerM12Fix (runs_11c_v5fix); "
                             "small: SmallMEC4 MeanPoolMLP (runs_small); "
                             "medium: MediumMEC4 GoalCrossAttention (runs_medium)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--blank-frac", type=float, default=BLANK_FRAC,
                        help="空候选合成比例（每 epoch 轮换；0=关闭）")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not os.environ.get("CITYNAV_DATA"):
        raise SystemExit("CITYNAV_DATA is not set")

    set_seed(args.seed)
    device = args.device
    run_dir = resolve_run_dir(args.model, args.seed, args.resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["classes"] = CLASSES
    config["model"] = MODEL_NAMES[args.model]
    config["data"] = "11B pure four-action teacher pool (11A final set, uncapped)"
    config["scheduler"] = "OneCycleLR per-batch, total_steps=epochs*steps_per_epoch"

    train_meta = json.loads((HERE.parent / "results/data/train_seen_memmap/meta.json")
                            .read_text(encoding="utf-8"))
    n_train = int(train_meta["n_states"])
    ep_ranges = json.loads((HERE.parent / "results/data/train_seen_memmap/ep_ranges.json")
                           .read_text(encoding="utf-8"))
    ep_ranges = {int(k): v for k, v in ep_ranges.items()}
    assert sum(e - s for s, e in ep_ranges.values()) == n_train, "ep_ranges mismatch"
    ep_list = sorted(ep_ranges)
    val_ds = ValMemmapDataset("val_seen")
    n_val = len(val_ds)
    steps_per_epoch = math.ceil(n_train / args.batch)
    print(f"train states {n_train} (memmap) | val states {n_val} | "
          f"steps/epoch {steps_per_epoch}", flush=True)

    model = MODEL_CLASSES[args.model]().to(device)
    encoder = FrozenEncoder(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    # v4 fix 3: per-batch OneCycleLR (standard) instead of per-epoch stepping
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
        pct_start=args.warmup, anneal_strategy="cos", cycle_momentum=False)
    start_epoch, start_step, best, global_step = 0, 0, -1.0, 0
    n_best_epochs = 0
    ckpt_path = run_dir / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best = ckpt.get("best", -1.0)
        global_step = ckpt.get("global_step", 0)
        if ckpt.get("epoch_done", True):
            start_epoch, start_step = ckpt["epoch"] + 1, 0
        else:
            start_epoch, start_step = ckpt["epoch"], ckpt["step"]
        restore_rng(ckpt["rng"])
        # n_best_epochs is NOT stored in the checkpoint; recompute it exactly
        # from history.jsonl so early-stop patience survives resume
        hpath = run_dir / "history.jsonl"
        if hpath.exists():
            f1s = []
            for line in hpath.read_text(encoding="utf-8").splitlines():
                try:
                    f1s.append(float(json.loads(line)["val_seen"]["macro_f1"]))
                except (json.JSONDecodeError, KeyError):
                    continue
            if f1s:
                best_i = int(np.argmax(f1s))
                n_best_epochs = len(f1s) - 1 - best_i
        print(f"resumed at epoch {start_epoch} step {start_step} (best={best:.4f} "
              f"n_best_epochs={n_best_epochs})", flush=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"params={model.n_params()} run={run_dir}", flush=True)

    nan_updates = 0
    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        # episode-block shuffle: states of one episode stay contiguous, so the
        # memmap reads are sequential on HDD (random per-state reads otherwise)
        rng = np.random.default_rng(args.seed * 1000 + epoch)
        ep_order = rng.permutation(ep_list)
        idx = np.concatenate([np.arange(s, e, dtype=np.int64)
                              for k in ep_order for s, e in [ep_ranges[k]]])
        blank = build_blank_mask(args.seed, epoch, ep_order, ep_ranges,
                                 args.blank_frac)
        first_batch = start_step if epoch == start_epoch else 0
        idx = idx[first_batch * args.batch:]
        blank = blank[first_batch * args.batch:]
        loader = DataLoader(ChunkedMemmapDataset("train_seen", idx,
                                                 blank_mask=blank),
                            batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate11,
                            persistent_workers=args.workers > 0,
                            prefetch_factor=4 if args.workers else None)
        model.train()
        total, count, correct, gnorm_sum, steps_done = 0.0, 0, 0, 0.0, 0
        for offset, batch in enumerate(loader):
            step = first_batch + offset
            logits = model(to_model_batch11(batch, encoder, device))
            label = torch.from_numpy(batch["label"]).to(device)
            loss = F.cross_entropy(logits, label)
            if not torch.isfinite(loss):
                nan_updates += 1
                if nan_updates <= 5:
                    print(f"  WARN non-finite loss at epoch {epoch} step {step}; "
                          f"skipping update (total {nan_updates})", flush=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            scheduler.step()          # v4 fix 3: per-batch schedule step
            global_step += 1
            steps_done += 1
            gnorm_sum += gnorm
            total += float(loss.item()) * len(label)
            correct += int((logits.argmax(-1) == label).sum())
            count += len(label)
            if steps_done % 1000 == 0 and steps_done:
                el = time.time() - started
                print(f"  epoch {epoch} batch {steps_done}/{steps_per_epoch} "
                      f"loss={total / count:.4f} acc={correct / count:.4f} "
                      f"{el:.0f}s", flush=True)
        train_loss = total / max(count, 1)
        train_acc = correct / max(count, 1)
        val = evaluate11(model, encoder, val_ds, device)
        epoch_metrics = {"epoch": epoch, "train_loss": train_loss,
                         "train_acc": train_acc, "train_steps": steps_done,
                         "grad_norm": gnorm_sum / max(steps_done, 1),
                         "val_seen": val, "lr": scheduler.get_last_lr()[0],
                         "seconds": round(time.time() - started, 1),
                         "global_step": global_step,
                         "gpu": gpu_snapshot(),
                         "resumed_from_step": first_batch,
                         "nan_updates": nan_updates}
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(epoch_metrics) + "\n")
        with (run_dir / "gpu_util.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"epoch": epoch, **epoch_metrics["gpu"]}) + "\n")
        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, 0, True,
                        config, best, epoch_metrics, global_step)
        torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                    "val": val, "train": {"loss": train_loss, "acc": train_acc}},
                   run_dir / f"epoch_{epoch:02d}.pt")
        improved = val["macro_f1"] > best
        if improved:
            best = val["macro_f1"]
            n_best_epochs = 0
            torch.save({"model": model.state_dict(), "config": config,
                        "epoch": epoch, "val": val, "macro_f1": best},
                       run_dir / "best.pt")
            print(f"  new best macro_f1={best:.4f} at epoch {epoch}", flush=True)
        else:
            n_best_epochs += 1
        print(json.dumps({"epoch": epoch, "loss": round(train_loss, 4),
                          "train_acc": round(train_acc, 4),
                          "val_loss": round(val["loss"], 4),
                          "val_acc": round(val["accuracy"], 4),
                          "val_macro_f1": round(val["macro_f1"], 4),
                          "val_stop_recall": round(val["per_class"]["Stop"]["recall"], 3),
                          "val_turn_recall": [round(val["per_class"]["TurnLeft"]["recall"], 3),
                                              round(val["per_class"]["TurnRight"]["recall"], 3)],
                          "per_class_f1": {c: round(val["per_class"][c]["f1"], 4)
                                           for c in CLASSES},
                          "pred_dist": {c: val["per_class"][c]["predicted"] for c in CLASSES},
                          "lr": round(scheduler.get_last_lr()[0], 7),
                          "grad_norm": round(epoch_metrics["grad_norm"], 4),
                          "best": round(best, 4),
                          "patience_left": args.patience - n_best_epochs,
                          "nan_updates": nan_updates,
                          "gpu": epoch_metrics["gpu"],
                          "sec": epoch_metrics["seconds"]}), flush=True)
        with (run_dir / "train_progress.json").open("w", encoding="utf-8") as fh:
            json.dump({"epoch": epoch, "epochs": args.epochs,
                       "global_step": global_step, "train_loss": train_loss,
                       "val_macro_f1": val["macro_f1"], "best": best,
                       "patience_left": args.patience - n_best_epochs,
                       "seconds": epoch_metrics["seconds"],
                       "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                      fh, indent=2)
        if n_best_epochs >= args.patience:
            print(f"early stop: no val macro_f1 improvement for "
                  f"{args.patience} epochs (best {best:.4f} at epoch "
                  f"{epoch - n_best_epochs})", flush=True)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
