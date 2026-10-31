#!/usr/bin/env python3
"""Train-Seen calibration for 09C: visual variance and observability gates.

Positives: teacher states with the target inside the view.
Hard negatives: states whose target is outside the view (the failure mode that
produced high-confidence wrong matches in 09A/09B), plus low-validity states.

Chooses:
  sigma_vis_m  robust RMS of the in-view localisation error
  tau2, margin2  score/margin thresholds maximising F1 with hard negatives
  agree_m      two-frame world-coordinate agreement (90th percentile of the
               distance between consecutive in-view matches)
  v_min        minimum raster validity at the matched patch
  d_max        Mahalanobis gate of the best mixture component (3 sigma)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/09_online_visual_goal_matching/scripts"))
sys.path.insert(0, str(ROOT / "experiments/04_visual_goal_global_map_pipeline/scripts"))
sys.path.insert(0, str(ROOT / "experiments/05_transformer_m_training/scripts"))

import matching  # noqa: E402
from continuous_belief import subpixel_match  # noqa: E402
from dataset_v2 import RasterStore  # noqa: E402

TEACHER = ROOT / "experiments/03b_target_directed_teacher/results"
RES = HERE.parent / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positives", type=int, default=800)
    parser.add_argument("--negatives", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    z = np.load(TEACHER / "teacher_train_seen.npz")
    keys = [int(k) for k in z["keys"]]
    off = z["offsets"]
    states, targets = z["states"], z["targets"]
    with (TEACHER / "teacher_train_seen.csv").open(encoding="utf-8") as fh:
        rows = {int(r["index"]): r for r in csv.DictReader(fh)}
    rng = np.random.default_rng(20261031)

    pos, neg = [], []
    eps_idx = rng.permutation(len(keys))
    for i in eps_idx:
        s = states[off[i]:off[i + 1]]
        t = targets[i]
        for j in range(len(s)):
            d = float(np.linalg.norm(s[j, :2] - t))
            col, row = matching.offset_to_pixel(t[0] - s[j, 0], t[1] - s[j, 1])
            in_view = (d <= 48.0 and 0 <= col <= 223 and 0 <= row <= 223)
            if in_view and len(pos) < args.positives and rng.random() < 0.02:
                pos.append((i, j))
            elif (not in_view) and d <= 120 and len(neg) < args.negatives and rng.random() < 0.01:
                neg.append((i, j))
        if len(pos) >= args.positives and len(neg) >= args.negatives:
            break
    print(f"positives {len(pos)} | hard negatives {len(neg)}")

    enc = matching.Encoder(args.device)
    rasters = RasterStore()
    goals, gfeat = {}, {}

    def goal(name, tx, ty):
        k = (name, round(float(tx), 2), round(float(ty), 2))
        if k not in goals:
            goals[k] = matching.prepare_goal(rasters.get(name), tx, ty)
        return k

    def ensure(ks):
        todo = [k for k in ks if k not in gfeat]
        if todo:
            out = enc.features(np.stack([goals[k] for k in todo]), ("layer2",))
            for i, k in enumerate(todo):
                gfeat[k] = out["layer2"][i]

    def measure(items):
        recs = []
        for i0 in range(0, len(items), args.batch):
            chunk = items[i0:i0 + args.batch]
            obs, metas = [], []
            for i, j in chunk:
                name = rows[keys[i]]["map"]
                p = states[off[i] + j]
                obs.append(matching.prepare_observation(rasters.get(name), p[0], p[1], p[2]))
                t = targets[i]
                metas.append((name, p, t, goal(name, t[0], t[1])))
            of = enc.features(np.stack(obs), ("layer2",))["layer2"]
            ensure([m[3] for m in metas])
            for n, (name, p, t, k) in enumerate(metas):
                m = subpixel_match(of[n], gfeat[k], 2)
                dx, dy = matching.pixel_to_offset(m["col"], m["row"])
                zx, zy = p[0] + dx, p[1] + dy
                err = float(math.hypot(zx - t[0], zy - t[1]))
                validity = rasters.get(name).valid_fraction(zx, zy, 8.0)
                recs.append({"err": err, "score": m["score"], "margin": m["margin"],
                             "z": [zx, zy], "validity": validity,
                             "border_px": min(m["col"], m["row"], 223 - m["col"],
                                              223 - m["row"]),
                             "episode": int(i)})
        return recs

    P = measure(pos)
    N = measure(neg)
    errs = np.array([r["err"] for r in P])
    good = errs <= 20.0
    sigma = float(np.sqrt(np.mean(errs[good] ** 2)))
    ps = np.array([r["score"] for r in P])
    pm = np.array([r["margin"] for r in P])
    ns = np.array([r["score"] for r in N])
    nm = np.array([r["margin"] for r in N])
    best = (-1, 0.8, 0.0)
    for tau in np.arange(0.70, 0.96, 0.02):
        for mt in np.arange(0.0, 0.16, 0.01):
            tp = int(np.sum((ps >= tau) & (pm >= mt) & good))
            fn = int(good.sum()) - tp
            fp = int(np.sum((ns >= tau) & (nm >= mt)))
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best[0]:
                best = (f1, float(tau), float(mt))
    # two-frame agreement: genuinely adjacent trajectory frames where both views
    # contain the target (a sparse positive sample cannot be used for this)
    gaps = []
    for i in rng.choice(len(keys), 80, replace=False):
        s_i = states[off[i]:off[i + 1]]
        t = targets[i]
        if len(s_i) < 2:
            continue
        cols, rows_ = matching.offset_to_pixel(t[0] - s_i[:, 0], t[1] - s_i[:, 1])
        d = np.linalg.norm(s_i[:, :2] - t, axis=1)
        inview = (d <= 48) & (cols >= 0) & (cols <= 223) & (rows_ >= 0) & (rows_ <= 223)
        pairs = [j for j in range(len(s_i) - 1) if inview[j] and inview[j + 1]]
        if not pairs:
            continue
        name = rows[keys[int(i)]]["map"]
        obs = [matching.prepare_observation(rasters.get(name), p[0], p[1], p[2]) for p in s_i]
        of = enc.features(np.stack(obs), ("layer2",))["layer2"]
        k = goal(name, t[0], t[1])
        ensure([k])
        zs = {}
        for j in pairs + [j + 1 for j in pairs]:
            if j in zs:
                continue
            m = subpixel_match(of[j], gfeat[k], 2)
            dx, dy = matching.pixel_to_offset(m["col"], m["row"])
            zs[j] = (s_i[j, 0] + dx, s_i[j, 1] + dy)
        for j in pairs:
            gaps.append(float(math.dist(zs[j], zs[j + 1])))
    agree = float(np.percentile(gaps, 90)) if gaps else 10.0
    v_min = float(np.percentile([r["validity"] for r in P], 5))
    out = {"source": "Train-Seen teacher states (seed 20261031)",
           "n_positives": len(P), "n_hard_negatives": len(N),
           "sigma_vis_m": sigma,
           "tau2": best[1], "margin2": best[2], "f1": best[0],
           "accept_rate_positives": float(np.mean((ps >= best[1]) & (pm >= best[2]))),
           "accept_rate_hard_negatives": float(np.mean((ns >= best[1]) & (nm >= best[2]))),
           "agree_m": agree, "v_min": v_min, "d_max": 3.0,
           "positive_median_err_m": float(np.median(errs)),
           "positive_mae_m": float(np.mean(errs)),
           "hard_negative_median_abs_target_err_m": float(
               np.median([r["err"] for r in N])) if N else None}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "09c_calibration.json").write_text(json.dumps(out, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
