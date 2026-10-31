#!/usr/bin/env python3
"""10A tau calibration for PQ/CA on Train-Seen only.

Same protocol as calibrate_09a.py:

* the identical 2,000 Train-Seen teacher states (calibration_subset(),
  rng seed 20261030, target within 48 m) -- held out from PQ/CA training;
* one same-map wrong target at least 100 m away per state (when possible),
  treated as a false-accept if it passes the gate;
* tau grid search maximising F1 of "localised within 10 m".

C1 keeps the frozen historical calibration (09a_calibration.json, tau 0.875).

Outputs: results/{method}/calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common10a  # noqa: E402
from common10a import RES, calibration_subset, load_teacher, require_data_root  # noqa: E402

from model10a import load_matcher  # noqa: E402

sys.path.insert(0, str(common10a.REPO / "experiments/09_online_visual_goal_matching/scripts"))
import geom  # noqa: E402
import matching  # noqa: E402


def main() -> int:
    require_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["pq", "ca"], required=True)
    parser.add_argument("--ckpt", default="", help="default: {method}/checkpoints/best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    from dataset_v2 import RasterStore  # local import

    ckpt = Path(args.ckpt) if args.ckpt else RES / args.method / "checkpoints" / "best.pt"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint missing: {ckpt} (train first)")
    model = load_matcher(args.method, str(ckpt), args.device)

    rows, keys, off, states, targets = load_teacher()
    picked = calibration_subset()

    # same-map wrong target >= 100 m away when possible (as calibrate_09a.py)
    by_map = {}
    for i, key in enumerate(keys):
        by_map.setdefault(rows[key]["map"], []).append(i)
    rng = np.random.default_rng(common10a.CALIB_SEED)
    wrong = []
    for i, _ in picked:
        pool = [j for j in by_map[rows[keys[i]]["map"]] if j != i]
        far = [j for j in pool if np.linalg.norm(targets[j] - targets[i]) > 100.0]
        wrong.append(int(rng.choice(far if far else pool)) if pool else None)

    encoder = matching.Encoder(args.device)
    rasters = RasterStore()
    goals: dict[tuple, np.ndarray] = {}
    gfeat: dict[tuple, np.ndarray] = {}

    def goal(name, tx, ty):
        k = (name, round(float(tx), 2), round(float(ty), 2))
        if k not in goals:
            goals[k] = matching.prepare_goal(rasters.get(name), tx, ty)
        return k

    def ensure(ks):
        todo = [k for k in ks if k not in gfeat]
        if todo:
            out = encoder.features(np.stack([goals[k] for k in todo]), ("layer2",))
            for i, k in enumerate(todo):
                gfeat[k] = out["layer2"][i]

    recs = []
    t0 = time.time()
    with torch.no_grad():
        for i0 in range(0, len(picked), args.batch):
            chunk = picked[i0:i0 + args.batch]
            obs, metas = [], []
            for i, j in chunk:
                name = rows[keys[i]]["map"]
                s = states[off[i] + j]
                t = targets[i]
                obs.append(matching.prepare_observation(rasters.get(name),
                                                        s[0], s[1], s[2]))
                kc = goal(name, t[0], t[1])
                w = wrong[i0 + len(metas)]
                kn = goal(rows[keys[w]]["map"], targets[w][0], targets[w][1]) \
                    if w is not None else kc
                metas.append((i, j, kc, kn, float(np.linalg.norm(s[:2] - t))))
            of = encoder.features(np.stack(obs), ("layer2",))["layer2"]
            ensure([m[2] for m in metas] + [m[3] for m in metas])
            of_t = torch.from_numpy(of).to(args.device)
            gf_t = torch.stack([torch.from_numpy(gfeat[k]).to(args.device)
                                for k in [m[2] for m in metas]])
            gn_t = torch.stack([torch.from_numpy(gfeat[k]).to(args.device)
                                for k in [m[3] for m in metas]])
            rc = model(of_t, gf_t)
            rn = model(of_t, gn_t)
            col, row, score, margin = (rc[k].cpu().numpy() for k in
                                       ("col", "row", "score", "margin"))
            ncol, nrow, nscore, nmargin = (rn[k].cpu().numpy() for k in
                                           ("col", "row", "score", "margin"))
            for n, (i, j, kc, kn, dist) in enumerate(metas):
                true_col, true_row = matching.offset_to_pixel(
                    targets[i][0] - states[off[i] + j][0],
                    targets[i][1] - states[off[i] + j][1])
                err = matching.localisation_error(float(col[n]), float(row[n]),
                                                  true_col, true_row)
                recs.append({"dist": dist, "err": err, "score": float(score[n]),
                             "margin": float(margin[n]), "neg_score": float(nscore[n]),
                             "neg_margin": float(nmargin[n])})

    tau_grid = np.arange(0.0, 1.001, 0.01)
    pos = np.array([max(r["err"], 0) <= 10.0 if np.isfinite(r["err"]) else False
                    for r in recs])
    sc = np.array([r["score"] for r in recs])
    nsc = np.array([r["neg_score"] for r in recs])
    best = (-1, None)
    for tau in tau_grid:
        acc = sc >= tau
        neg_acc = nsc >= tau
        tp = int(np.sum(pos & acc))
        fn = int(np.sum(pos)) - tp
        fp = int(np.sum(~pos & acc)) + int(np.sum(neg_acc))
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[0]:
            best = (f1, float(tau))
    err = np.array([r["err"] for r in recs if np.isfinite(r["err"])])
    out = {"method": args.method, "checkpoint": str(ckpt),
           "source": "teacher_train_seen 2000-state calibration subset "
           "(seed 20261030, target <= 48 m; excluded from training)",
           "n_states": len(recs), "tau": best[1], "f1": round(best[0], 4),
           "neg_goal_false_accept_at_tau": round(float(np.mean(nsc >= best[1])), 4),
           "positive_rate": round(float(np.mean(pos)), 4),
           "err_mae_m": round(float(np.mean(err)), 3),
           "err_median_m": round(float(np.median(err)), 3),
           "err_p90_m": round(float(np.percentile(err, 90)), 3),
           "prec_5m": round(float(np.mean(err <= 5)), 4),
           "prec_10m": round(float(np.mean(err <= 10)), 4),
           "prec_20m": round(float(np.mean(err <= 20)), 4),
           "seconds": round(time.time() - t0, 1),
           "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (RES / args.method).mkdir(parents=True, exist_ok=True)
    (RES / args.method / "calibration.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
