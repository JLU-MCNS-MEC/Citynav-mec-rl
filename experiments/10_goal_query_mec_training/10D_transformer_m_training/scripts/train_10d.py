#!/usr/bin/env python3
"""10D: Transformer-M v3 training on the 10C input contract (seed 42, once).

* data:    10C cache — Train-Seen teacher decision states (12/episode,
           5-bin stratified, global subsample) + C1 recovery states
           (DAgger-style, <=25% of the mix).  Plain four-class cross entropy,
           NO class weights / pos_weight / focal loss;
* model:   TransformerM v3 (10C contract): current 49x2048 layer4 spatial,
           C1 goal desc 512, 16-step causal history, 10-d belief + 5-d match;
* loop:    OneCycleLR like Stage 05; resumable checkpoint.pt (model/opt/sched/
           RNG/batch position); cache identity re-checked on resume;
* eval:    fixed Val-Seen subset (400 episodes x 4 states) per epoch with
           loss, accuracy, Macro-F1, per-class P/R/F1, confusion matrix,
           STOP/turn recall; best.pt by Val Macro-F1; early stop patience 10;
* outputs: runs_10d/seed42/{best.pt, epoch_NN.pt, checkpoint.pt,
           history.jsonl, config.json, train_progress.json, gpu_util.jsonl}.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
assert (REPO / "experiments").is_dir(), f"REPO 解析错误: {REPO}"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments/05_transformer_m_training/scripts"))
sys.path.insert(0, str(REPO / "experiments/04_visual_goal_global_map_pipeline/scripts"))

from dataset_v2 import FrozenEncoder, RasterStore, history_block  # noqa: E402
from model10c import TransformerM  # noqa: E402

CACHE = REPO / "experiments/10_goal_query_mec_training/10C_mec_input_dataset/results/cache"
RUNS = HERE.parent / "results" / "runs_10d"
CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]
CACHE_VERSION = "10c-c1belief-v1"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class RamDataset(Dataset):
    """One sample = one decision state, served from RAM materialised arrays
    (materialize_10d.py); the only per-sample IO is the current-image crop."""

    def __init__(self, z, idx):
        self.z = z
        self.idx = idx
        self.rasters: RasterStore | None = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i: int) -> dict:
        if self.rasters is None:      # one raster cache per worker process
            self.rasters = RasterStore()
        k = self.idx[i]
        x, y, yaw = (float(v) for v in self.z["states"][k])
        image = self.rasters.get(self.z["maps"][k]).heading_crop(x, y, yaw)
        return {"image": image,
                "target_emb": self.z["target_emb"][k],
                "belief": self.z["belief"][k],
                "match": self.z["match"][k],
                "history_vis": self.z["hist_vis"][k].astype(np.float32),
                "history_pose": self.z["hist_pose"][k],
                "history_action": np.eye(4, dtype=np.float32)[self.z["hist_act"][k]],
                "history_mask": self.z["hist_mask"][k],
                "label": int(self.z["labels"][k])}


def collate(samples: list[dict]) -> dict:
    return {
        "image": np.stack([s["image"] for s in samples]),
        "target_emb": np.stack([s["target_emb"] for s in samples]),
        "history_vis": np.stack([s["history_vis"] for s in samples]),
        "history_pose": np.stack([s["history_pose"] for s in samples]),
        "history_action": np.stack([s["history_action"] for s in samples]),
        "history_mask": np.stack([s["history_mask"] for s in samples]),
        "belief": np.stack([s["belief"] for s in samples]),
        "match": np.stack([s["match"] for s in samples]),
        "label": np.asarray([s["label"] for s in samples], np.int64),
    }


def to_model_batch(batch: dict, encoder: FrozenEncoder, device: str) -> dict:
    out = {"current": encoder(batch["image"])}
    for key in ("target_emb", "history_vis", "history_pose", "history_action",
                "belief", "match"):
        out[key] = torch.from_numpy(batch[key]).to(device)
    out["history_mask"] = torch.from_numpy(batch["history_mask"]).to(device)
    return out


def metrics(pred: np.ndarray, label: np.ndarray, loss: float | None = None) -> dict:
    out = {"accuracy": float((pred == label).mean()) if len(label) else 0.0,
           "n": int(len(label))}
    if loss is not None:
        out["loss"] = float(loss)
    per_class = {}
    for c, name in enumerate(CLASSES):
        tp = int(((pred == c) & (label == c)).sum())
        fp = int(((pred == c) & (label != c)).sum())
        fn = int(((pred != c) & (label == c)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1,
                           "support": int((label == c).sum()),
                           "predicted": int((pred == c).sum())}
    out["per_class"] = per_class
    out["macro_f1"] = float(np.mean([v["f1"] for v in per_class.values()]))
    conf = np.zeros((4, 4), int)
    for p, l in zip(pred, label):
        conf[l, p] += 1
    out["confusion"] = conf.tolist()
    return out


@torch.no_grad()
def evaluate(model, encoder, dataset, device, batch_size=128) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate)
    preds, labels, losses = [], [], []
    for batch in loader:
        logits = model(to_model_batch(batch, encoder, device))
        label = torch.from_numpy(batch["label"]).to(device)
        losses.append(float(F.cross_entropy(logits, label).item()) * len(label))
        preds.extend(logits.argmax(-1).cpu().numpy().tolist())
        labels.extend(batch["label"].tolist())
    mean_loss = float(np.sum(losses) / max(len(labels), 1))
    return metrics(np.asarray(preds), np.asarray(labels), mean_loss)


def gpu_snapshot() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        util, mem, temp = out.stdout.strip().split(",")
        return {"gpu_util_pct": int(util.strip().split()[0]),
                "mem_mib": int(mem.strip().split()[0]),
                "temp_c": int(temp.strip().split()[0])}
    except Exception as e:
        return {"error": str(e)[:100]}


def save_checkpoint(path, model, optimizer, scheduler, epoch, step, epoch_done,
                    config, best, metrics_epoch, global_step) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch, "step": step, "epoch_done": epoch_done,
                "global_step": int(global_step), "config": config,
                "best": best, "metrics": metrics_epoch, "rng": rng_state()}, tmp)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not os.environ.get("CITYNAV_DATA"):
        raise SystemExit("CITYNAV_DATA is not set")

    set_seed(args.seed)
    device = args.device
    run_dir = RUNS / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["classes"] = CLASSES

    # ---- load the 10C materialised arrays ----------------------------------
    def load_ram(name: str) -> dict:
        d = HERE.parent / "results" / name
        if not d.is_dir():
            raise SystemExit(f"missing {d} - run materialize_10d.py first")
        z = {f.stem: np.load(f, allow_pickle=(f.stem == "maps"))
             for f in sorted(d.glob("*.npy"))}
        return z

    train_z = load_ram("train_combined")
    val_z = load_ram("val")
    n_train = int(len(train_z["labels"]))
    n_val = int(len(val_z["labels"]))
    print(f"train states {n_train} (RAM) | val states {n_val} (RAM)", flush=True)

    model = TransformerM().to(device)
    encoder = FrozenEncoder(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs,
        pct_start=args.warmup, anneal_strategy="cos", cycle_momentum=False)
    start_epoch, start_step, best, global_step = 0, 0, -1.0, 0
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
        print(f"resumed at epoch {start_epoch} step {start_step} (best={best:.4f})",
              flush=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"params={model.n_params()} run={run_dir}", flush=True)

    val_ds = RamDataset(val_z, np.arange(n_val))
    n_best_epochs = 0
    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        rng = np.random.default_rng(args.seed * 1000 + epoch)
        idx = np.arange(n_train)
        rng.shuffle(idx)
        first_batch = start_step if epoch == start_epoch else 0
        idx = idx[first_batch * args.batch:]
        loader = DataLoader(RamDataset(train_z, idx), batch_size=args.batch,
                            shuffle=False, num_workers=args.workers,
                            collate_fn=collate,
                            persistent_workers=args.workers > 0,
                            prefetch_factor=4 if args.workers else None)
        model.train()
        total, count, correct, gnorm_sum, steps_done = 0.0, 0, 0, 0.0, 0
        for offset, batch in enumerate(loader):
            step = first_batch + offset
            logits = model(to_model_batch(batch, encoder, device))
            label = torch.from_numpy(batch["label"]).to(device)
            loss = F.cross_entropy(logits, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            global_step += 1
            steps_done += 1
            gnorm_sum += gnorm
            total += float(loss.item()) * len(label)
            correct += int((logits.argmax(-1) == label).sum())
            count += len(label)
        scheduler.step()
        train_loss = total / max(count, 1)
        train_acc = correct / max(count, 1)
        val = evaluate(model, encoder, val_ds, device)
        epoch_metrics = {"epoch": epoch, "train_loss": train_loss,
                         "train_acc": train_acc, "train_steps": steps_done,
                         "grad_norm": gnorm_sum / max(steps_done, 1),
                         "val_seen": val, "lr": scheduler.get_last_lr()[0],
                         "seconds": round(time.time() - started, 1),
                         "global_step": global_step,
                         "gpu": gpu_snapshot(),
                         "resumed_from_step": first_batch}
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
                          "pred_dist": {c: val["per_class"][c]["predicted"] for c in CLASSES},
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
