#!/usr/bin/env python3
"""Stage 05: Transformer-M training on the 03B direct-teacher labels.

* episode-level sampling: each epoch draws ``k`` decision states per episode;
  history and belief come from the episode cache, the current 7x7x2048
  observation is encoded on the fly by the frozen ResNet;
* plain four-class cross entropy - no class weights, pos_weight or focal loss;
* checkpoint.pt every ``--save-every`` batches and at every epoch end, holding
  model / optimizer / scheduler / RNG state and the exact batch position, so a
  killed run resumes inside the epoch instead of restarting it;
* the cache identity (candidate library, encoder, cache version) is stored in
  the checkpoint and re-checked on resume: a checkpoint trained on another
  candidate grid is refused, never silently continued.

Usage:
  python train_v2.py --seed 42 --epochs 30 --resume
  python train_v2.py --seed 42 --episodes 200 --epochs 3 --tag overfit200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset_v2 import CACHE, CANDS, CACHE_VERSION, EpisodeStore, FrozenEncoder, RasterStore  # noqa: E402
from model_v2 import TransformerM  # noqa: E402

RUNS = HERE.parent / "runs_v2"
CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]


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


def sample_pairs(store: EpisodeStore, k: int, rng: np.random.Generator,
                 episodes: np.ndarray | None = None) -> list[tuple[int, int]]:
    ids = np.arange(len(store)) if episodes is None else episodes
    pairs = []
    for e in ids:
        t_len = store.length(int(e))
        kk = min(k, t_len)
        for t in rng.choice(t_len, size=kk, replace=False):
            pairs.append((int(e), int(t)))
    return pairs


def fixed_pairs(store: EpisodeStore, k: int, seed: int,
                episodes: np.ndarray | None = None) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    return sample_pairs(store, k, rng, episodes)


class SampleDataset(Dataset):
    def __init__(self, store: EpisodeStore, pairs: list[tuple[int, int]]):
        self.store = store
        self.pairs = pairs
        self.rasters: RasterStore | None = None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict:
        if self.rasters is None:      # one raster cache per worker process
            self.rasters = RasterStore()
        e, t = self.pairs[i]
        return self.store.sample(e, t, self.rasters)


def collate(samples: list[dict]) -> dict:
    return {
        "image": np.stack([s["image"] for s in samples]),
        "target_emb": np.stack([s["target_emb"] for s in samples]),
        "history_vis": np.stack([s["history_vis"] for s in samples]),
        "history_pose": np.stack([s["history_pose"] for s in samples]),
        "history_action": np.stack([s["history_action"] for s in samples]),
        "history_mask": np.stack([s["history_mask"] for s in samples]),
        "belief": np.stack([s["belief"] for s in samples]),
        "label": np.asarray([s["label"] for s in samples], np.int64),
    }


def to_model_batch(batch: dict, encoder: FrozenEncoder, device: str) -> dict:
    out = {}
    # uint8 HWC over the bus; float32 / 255 / ImageNet mean-std happen on the GPU
    out["current"] = encoder(batch["image"])
    for key in ("target_emb", "history_vis", "history_pose", "history_action", "belief"):
        out[key] = torch.from_numpy(batch[key]).to(device)
    out["history_mask"] = torch.from_numpy(batch["history_mask"]).to(device)
    return out


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
def evaluate(model, encoder, store: EpisodeStore, pairs, device, batch_size=128) -> dict:
    model.eval()
    loader = DataLoader(SampleDataset(store, pairs), batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate)
    preds, labels = [], []
    for batch in loader:
        logits = model(to_model_batch(batch, encoder, device))
        preds.extend(logits.argmax(-1).cpu().numpy().tolist())
        labels.extend(batch["label"].tolist())
    return metrics(np.asarray(preds), np.asarray(labels))


def cache_identity(index: dict) -> dict:
    return {"candidate_library": index["candidate_library"],
            "candidate_version": index.get("candidate_version"),
            "cache_version": index["cache_version"],
            "encoder": index["encoder"], "states": index["states"],
            "episodes": index["episodes_total"]}


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, step: int,
                    epoch_done: bool, config: dict, best: float, metrics_epoch: dict | None,
                    cache_id: dict, global_step: int) -> None:
    """Write the resume state atomically (a kill during torch.save must not
    leave a truncated checkpoint behind)."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch, "step": step, "epoch_done": epoch_done,
                "global_step": int(global_step),
                "config": config, "best": best, "metrics": metrics_epoch,
                "rng": rng_state(), "cache": cache_id}, tmp)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--k-per-episode", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.03)
    parser.add_argument("--episodes", type=int, default=0,
                        help="limit Train-Seen episodes (overfit diagnostic)")
    parser.add_argument("--eval-episodes", type=int, default=200,
                        help="episodes used for the fixed per-epoch evaluation")
    parser.add_argument("--eval-per-episode", type=int, default=2,
                        help="states per episode in the fixed evaluation set")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tag", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=100,
                        help="mid-epoch checkpoint interval in batches (0 disables)")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device
    run_dir = RUNS / (args.tag or f"seed{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["classes"] = CLASSES

    train_store = EpisodeStore("train_seen", CACHE)
    val_store = EpisodeStore("val_seen", CACHE)
    unseen_store = EpisodeStore("val_unseen", CACHE)
    cache_id = cache_identity(train_store.index)
    if cache_id["candidate_version"] != CANDS.name:
        raise RuntimeError(f"cache {cache_id['candidate_version']} != {CANDS.name}")
    config.update({"candidate_library": CANDS.name, "cache": cache_id})

    train_ids = np.arange(len(train_store))
    if args.episodes:
        train_ids = train_ids[:args.episodes]
    eval_train = fixed_pairs(train_store, args.eval_per_episode, seed=7,
                             episodes=train_ids[:args.eval_episodes])
    eval_val = fixed_pairs(val_store, args.eval_per_episode, seed=7,
                           episodes=np.arange(len(val_store))[:args.eval_episodes])
    eval_unseen = fixed_pairs(unseen_store, args.eval_per_episode, seed=7,
                              episodes=np.arange(len(unseen_store))[:args.eval_episodes])

    model = TransformerM().to(device)
    encoder = FrozenEncoder(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs,
        pct_start=args.warmup, anneal_strategy="cos", cycle_momentum=False)
    start_epoch, start_step, best, global_step = 0, 0, -1.0, 0
    log_path = run_dir / "history.jsonl"
    ckpt_path = run_dir / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved = ckpt.get("cache", {})
        if saved.get("candidate_library") != cache_id["candidate_library"] or \
                saved.get("cache_version") != cache_id["cache_version"] or \
                saved.get("encoder") != cache_id["encoder"]:
            raise RuntimeError(
                "checkpoint was trained on a different cache:\n"
                f"  checkpoint: {saved}\n  current:    {cache_id}\n"
                "refusing to resume - move the old run dir away and start fresh")
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
        print(f"resumed at epoch {start_epoch} step {start_step} (best={best:.4f}, "
              f"global_step={global_step})", flush=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"params={model.n_params()} run={run_dir} train_episodes={len(train_ids)} "
          f"cache={cache_id['candidate_version']} states={cache_id['states']}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        rng = np.random.default_rng(args.seed * 1000 + epoch)
        pairs = sample_pairs(train_store, args.k_per_episode, rng, train_ids)
        rng.shuffle(pairs)
        first_batch = start_step if epoch == start_epoch else 0
        pairs = pairs[first_batch * args.batch:]
        dataset = SampleDataset(train_store, pairs)
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate,
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
            if args.save_every and (step + 1) % args.save_every == 0:
                save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, step + 1,
                                False, config, best, None, cache_id, global_step)
            if args.max_steps and offset + 1 >= args.max_steps:
                break
        scheduler.step()
        train_loss = total / max(count, 1)
        train_acc = correct / max(count, 1)
        val = evaluate(model, encoder, val_store, eval_val, device)
        train_eval = evaluate(model, encoder, train_store, eval_train, device)
        epoch_metrics = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "train_steps": steps_done, "grad_norm": gnorm_sum / max(steps_done, 1),
                         "train_subset": train_eval, "val_seen": val,
                         "lr": scheduler.get_last_lr()[0],
                         "seconds": round(time.time() - started, 1),
                         "global_step": global_step,
                         "resumed_from_step": first_batch}
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            epoch_metrics["val_unseen"] = evaluate(model, encoder, unseen_store, eval_unseen, device)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(epoch_metrics) + "\n")
        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, 0, True,
                        config, best, epoch_metrics, cache_id, global_step)
        torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                    "val": val, "train": {"loss": train_loss, "acc": train_acc},
                    "grad_norm": epoch_metrics["grad_norm"]},
                   run_dir / f"epoch_{epoch:02d}.pt")
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                        "val": val, "macro_f1": best}, run_dir / "best.pt")
            print(f"  new best macro_f1={best:.4f} at epoch {epoch}", flush=True)
        print(json.dumps({"epoch": epoch, "loss": round(train_loss, 4),
                          "train_acc": round(train_acc, 4),
                          "val_acc": round(val["accuracy"], 4),
                          "val_macro_f1": round(val["macro_f1"], 4),
                          "val_stop_recall": round(val["per_class"]["Stop"]["recall"], 3),
                          "val_turn_recall": [round(val["per_class"]["TurnLeft"]["recall"], 3),
                                              round(val["per_class"]["TurnRight"]["recall"], 3)],
                          "pred_dist": {c: val["per_class"][c]["predicted"] for c in CLASSES},
                          "grad_norm": round(epoch_metrics["grad_norm"], 3),
                          "sec": epoch_metrics["seconds"]}), flush=True)
        with (run_dir / "train_progress.json").open("w", encoding="utf-8") as fh:
            json.dump({"epoch": epoch, "epochs": args.epochs, "global_step": global_step,
                       "train_loss": train_loss, "val_macro_f1": val["macro_f1"],
                       "val_acc": val["accuracy"],
                       "stop_recall": val["per_class"]["Stop"]["recall"],
                       "seconds": epoch_metrics["seconds"],
                       "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
