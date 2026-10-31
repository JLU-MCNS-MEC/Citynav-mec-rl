#!/usr/bin/env python3
"""10A training v2: fair PQ / CA matchers on the C1 mechanism (seed 42).

* data:    Train-Seen teacher state image cache (build_cache10a.py), with the
           2,000-state calibration subset excluded;
* encoder: frozen ResNet-50 layer2 features (matching.Encoder, fp32), on the
           fly, never fine-tuned;
* loss:    map-level supervision only.  In-view states: Gaussian blob
           (sigma=1.5 cells) centred at the true cell of the 28x28 match map;
           out-of-view states: all-low target (PQ: -1 cosine map / CA: zero
           logits), which trains the peak score to stay low when the target
           is outside the view -- the C1 gate semantics;
* PQ:      MSE on the cosine map; CA: BCE on the pre-softmax attention logits;
* position extraction, score, margin: identical to C1 (see model10a.py);
* resume:  --resume loads last.pt (model + optimizer + epoch); best.pt tracks
           the lowest training loss;
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common10a  # noqa: E402
from common10a import RES, require_data_root  # noqa: E402

from model10a import CrossAttnMatcher, PointQueryMatcher, n_params  # noqa: E402

sys.path.insert(0, str(common10a.REPO / "experiments/09_online_visual_goal_matching/scripts"))
import matching  # noqa: E402

CACHE = RES / "cache"
SCALE = common10a.SCALE
CELL = common10a.OUT / 28.0          # 8 px per map cell
BATCH = 128
LR = 1e-3
EPOCHS = 12
SEED = 42
BLOB_SIGMA = 1.5


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
        st = torch.load(last_pt, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_epoch = st["epoch"] + 1
        best_loss = st.get("best_loss", float("inf"))
        history = st.get("history", [])
        print(f"resumed from {last_pt} at epoch {start_epoch} "
              f"(best_loss {best_loss:.4f})", flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        "method": args.method, "version": "v2-fair-on-C1", "seed": args.seed,
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "params": n_params(model),
        "encoder": "frozen resnet50 IMAGENET1K_V2 layer2 fp32",
        "fairness_contract": "goal desc = C1 central 2x2 block; obs map = C1 "
        "2x2-neighbourhood pooling; position = C1 peak + 3x3 subpixel centroid "
        "+ margin suppression; PQ = learnable query/value projection (identity "
        "init reduces to C1 exactly, verified 1.2e-7); CA = attention-weight "
        "map replaces the cosine map; score = map peak (C1 semantics)",
        "loss": "map-level: in-view Gaussian blob (sigma 1.5 cells) at true "
        "cell; out-of-view all-low. PQ: MSE on cosine map [-1,1]; "
        "CA: BCE on pre-softmax logits",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2) + "\n",
        encoding="utf-8")

    ds = StateDataset(rows)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=args.workers, drop_last=False,
                                     persistent_workers=True)
    grid = torch.arange(28, device=device, dtype=torch.float32)
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
            b = of_t.shape[0]
            cell_col = target[:, 0] / CELL
            cell_row = target[:, 1] / CELL
            dx = grid.view(1, 1, -1) - cell_col.view(b, 1, 1)
            dy = grid.view(1, -1, 1) - cell_row.view(b, 1, 1)
            blob = torch.exp(-(dx ** 2 + dy ** 2) / (2 * BLOB_SIGMA ** 2))
            blob = blob * in_view.view(b, 1, 1).float()
            if args.method == "pq":
                m = model.match_map(of_t, gf_t)
                loss = F.mse_loss(m, 2 * blob - 1)
            else:
                logits = model.match_logits(of_t, gf_t)
                loss = F.binary_cross_entropy_with_logits(logits, blob)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                r = model(of_t, gf_t)
                err_m = torch.hypot(r["col"] - target[:, 0],
                                    r["row"] - target[:, 1]) * SCALE
                tot[0] += float(loss.item()) * b
                iv = in_view.bool()
                tot[1] += float((err_m[iv] <= 5).sum()) if iv.any() else 0.0
                tot[2] += float((err_m[iv] <= 10).sum()) if iv.any() else 0.0
                tot[3] += float(iv.sum())
            n += b
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
