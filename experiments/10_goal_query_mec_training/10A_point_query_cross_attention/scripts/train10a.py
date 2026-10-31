#!/usr/bin/env python3
"""10A training: Point Query / Cross-Attention matcher heads (seed 42).

* data:    Train-Seen teacher state image cache (build_cache10a.py), with the
           2,000-state calibration subset excluded;
* encoder: frozen ResNet-50 layer2 features (matching.Encoder, fp32), on the
           fly, never fine-tuned;
* loss:    smooth-L1 on (col,row) for every proposal, masked to states whose
           target is inside the 100 m view (out-of-view offsets are
           unrepresentable in the bounded head); BCE on the per-proposal
           confidence whose label is "predicted offset within 10 m";
* resume:  --resume loads last.pt (model + optimizer + epoch); best.pt is
           tracked by the lowest validation-free training loss;
* outputs: results/{method}/checkpoints/{best,last,epoch_NN}.pt,
           history.jsonl, config.json (all atomic).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common10a  # noqa: E402
from common10a import RES, require_data_root  # noqa: E402

from model10a import CrossAttnMatcher, PointQueryMatcher, n_params  # noqa: E402

sys.path.insert(0, str(common10a.REPO / "experiments/09_online_visual_goal_matching/scripts"))
import matching  # noqa: E402

CACHE = RES / "cache"
SCALE = common10a.SCALE
BATCH = 128
LR = 1e-3
EPOCHS = 12
SEED = 42


class StateDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        obs = np.load(CACHE / r["obs_file"])
        goal = np.load(CACHE / r["goal_file"])
        true_col = float(r["true_col"])
        true_row = float(r["true_row"])
        in_view = int(r["in_view"])
        return obs, goal, np.array([true_col, true_row], np.float32), in_view


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    require_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["pq", "ca"], required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    out_dir = RES / args.method
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with (CACHE / "manifest.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"train states: {len(rows)} | in_view: "
          f"{sum(int(r['in_view']) for r in rows)}", flush=True)

    device = args.device
    model = PointQueryMatcher() if args.method == "pq" else CrossAttnMatcher()
    model.to(device)
    encoder = matching.Encoder(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_epoch = 0
    best_loss = float("inf")
    history: list[dict] = []

    last_pt = ckpt_dir / "last.pt"
    if args.resume and last_pt.exists():
        st = torch.load(last_pt, map_location=device, weights_only=True)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_epoch = st["epoch"] + 1
        best_loss = st.get("best_loss", float("inf"))
        history = st.get("history", [])
        print(f"resumed from {last_pt} at epoch {start_epoch} "
              f"(best_loss {best_loss:.4f})", flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        "method": args.method, "seed": args.seed, "epochs": args.epochs,
        "batch": args.batch, "lr": args.lr, "params": n_params(model),
        "cache_manifest_sha": None, "encoder": "frozen resnet50 IMAGENET1K_V2 "
        "layer2 fp32", "loss": "smoothL1(col,row) masked in-view + BCE(score, "
        "pred-within-10m)", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
        indent=2) + "\n", encoding="utf-8")

    ds = StateDataset(rows)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=args.workers, drop_last=False,
                                     persistent_workers=True)
    reg_fn = torch.nn.SmoothL1Loss(reduction="none")
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    # score logits: use the pre-sigmoid score for BCE
    step = 0
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tot = np.zeros(4)
        n = 0
        t0 = time.time()
        for obs, goal, target, in_view in dl:
            obs, goal = obs.numpy(), goal.numpy()
            target = target.to(device)
            in_view = in_view.to(device)
            with torch.no_grad():
                of = encoder.features(obs, ("layer2",))["layer2"]
                gf = encoder.features(goal, ("layer2",))["layer2"]
            of_t = torch.from_numpy(of).to(device)
            gf_t = torch.from_numpy(gf).to(device)
            res = model(of_t, gf_t)
            out = res["out"]                       # (B,N,3) col,row,score_logit
            err_px = torch.hypot(out[..., 0] - target[:, 0][:, None],
                                 out[..., 1] - target[:, 1][:, None])
            reg = reg_fn(out[..., 0], target[:, 0][:, None].expand_as(out[..., 0])) \
                + reg_fn(out[..., 1], target[:, 1][:, None].expand_as(out[..., 1]))
            reg = reg * in_view[:, None].float()
            err_m = err_px * SCALE
            label = (err_m <= 10.0).float().detach()       # per-proposal quality
            b = bce(out[..., 2], label)
            loss = reg.mean() + b.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                top_err = err_m.gather(1, out[..., 2].max(1).indices.unsqueeze(1)
                                       ).squeeze(1)
                tot[0] += float(loss.item()) * len(obs)
                tot[1] += float((top_err[in_view.bool()] <= 5).sum()) \
                    if in_view.any() else 0.0
                tot[2] += float((top_err[in_view.bool()] <= 10).sum()) \
                    if in_view.any() else 0.0
                tot[3] += float(in_view.sum())
            n += len(obs)
            step += 1
            if step % 50 == 0:
                print(f"e{epoch} step {step} loss {loss.item():.4f} | "
                      f"{time.time() - t0:.0f}s", flush=True)
        train_loss = tot[0] / n
        p5 = tot[1] / max(tot[3], 1)
        p10 = tot[2] / max(tot[3], 1)
        hist = {"epoch": epoch, "loss": round(train_loss, 5),
                "in_view_prec5": round(float(p5), 4),
                "in_view_prec10": round(float(p10), 4),
                "seconds": round(time.time() - t0, 1)}
        history.append(hist)
        with (out_dir / "history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(hist) + "\n")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "best_loss": best_loss, "history": history,
                    "seed": args.seed}, ckpt_dir / "last.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "best_loss": best_loss, "history": history,
                    "seed": args.seed}, ckpt_dir / f"epoch_{epoch:02d}.pt")
        if train_loss < best_loss:
            best_loss = float(train_loss)
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "loss": best_loss, "seed": args.seed},
                       ckpt_dir / "best.pt")
        print(f"epoch {epoch} loss {train_loss:.4f} (best {best_loss:.4f}) | "
              f"in-view P5 {p5:.3f} P10 {p10:.3f} | {time.time() - t0:.0f}s "
              f"| total {time.time() - started:.0f}s", flush=True)
    print(f"training finished: {args.method} | total {time.time() - started:.0f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
