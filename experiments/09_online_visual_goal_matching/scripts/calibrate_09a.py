#!/usr/bin/env python3
"""Threshold calibration on Train-Seen only (09A).

Samples teacher Train-Seen states (never used for Stage 05 model selection),
matches the correct goal and a same-map wrong goal, and grid-searches the
confidence / margin thresholds that maximise F1 of "localised within 10 m".

The resulting thresholds are written to 09a_calibration.json and frozen; the
validation splits are only used for evaluation afterwards.
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
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "04_visual_goal_global_map_pipeline/scripts"))
sys.path.insert(0, str(ROOT / "05_transformer_m_training/scripts"))

import geom  # noqa: E402
import matching  # noqa: E402
from dataset_v2 import RasterStore  # noqa: E402

TEACHER = ROOT / "03b_target_directed_teacher/results"
RES = HERE.parent / "results"
METHODS = ("rgb_ncc", "layer3", "layer2")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--out", default=str(RES / "09a_calibration.json"))
    args = parser.parse_args()
    z = np.load(TEACHER / "teacher_train_seen.npz")
    keys = [int(k) for k in z["keys"]]
    off = z["offsets"]
    states, targets = z["states"], z["targets"]
    with (TEACHER / "teacher_train_seen.csv").open(encoding="utf-8") as fh:
        maps = {int(r["index"]): r["map"] for r in csv.DictReader(fh)}
    rng = np.random.default_rng(20261030)
    # candidate states: target within 48 m (calibration of the positive class)
    cand = []
    for i, key in enumerate(keys):
        s = states[off[i]:off[i + 1]]
        t = targets[i]
        d = np.linalg.norm(s[:, :2] - t, axis=1)
        for j in np.where(d <= 48)[0]:
            cand.append((i, int(j)))
    print(f"train candidate states: {len(cand)}")
    take = rng.choice(len(cand), size=min(args.n, len(cand)), replace=False)
    picked = [cand[i] for i in take]
    # a same-map wrong target at least 100 m away when possible
    by_map = {}
    for i, key in enumerate(keys):
        by_map.setdefault(maps[key], []).append(i)
    wrong = []
    for i, _ in picked:
        pool = [j for j in by_map[maps[keys[i]]] if j != i]
        far = [j for j in pool
               if np.linalg.norm(targets[j] - targets[i]) > 100.0]
        wrong.append(int(rng.choice(far if far else pool)) if pool else None)
    encoder = matching.Encoder(args.device)
    rasters = RasterStore()
    goals: dict[tuple, np.ndarray] = {}
    gfeat: dict[tuple, dict] = {}

    def goal(name, tx, ty):
        k = (name, round(float(tx), 2), round(float(ty), 2))
        if k not in goals:
            goals[k] = matching.prepare_goal(rasters.get(name), tx, ty)
        return k

    def ensure(ks):
        todo = [k for k in ks if k not in gfeat]
        if todo:
            out = encoder.features(np.stack([goals[k] for k in todo]), ("layer2", "layer3"))
            for i, k in enumerate(todo):
                gfeat[k] = {"layer2": out["layer2"][i], "layer3": out["layer3"][i]}

    recs = []
    for i0 in range(0, len(picked), args.batch):
        chunk = picked[i0:i0 + args.batch]
        obs, metas = [], []
        for i, j in chunk:
            name = maps[keys[i]]
            s = states[off[i] + j]
            t = targets[i]
            obs.append(matching.prepare_observation(rasters.get(name), s[0], s[1], s[2]))
            kc = goal(name, t[0], t[1])
            w = wrong[i0 + len(metas)]
            kn = goal(maps[keys[w]], targets[w][0], targets[w][1]) if w is not None else kc
            metas.append((i, j, kc, kn, float(np.linalg.norm(s[:2] - t))))
        of = encoder.features(np.stack(obs), ("layer2", "layer3"))
        ensure([m[2] for m in metas] + [m[3] for m in metas])
        for n, (i, j, kc, kn, dist) in enumerate(metas):
            ofi = {lay: of[lay][n] for lay in ("layer2", "layer3")}
            for meth in METHODS:
                if meth == "rgb_ncc":
                    pc = matching.match_rgb(obs[n], goals[kc]) or (np.nan, np.nan, 0.0, 0.0)
                    pn = matching.match_rgb(obs[n], goals[kn]) or (np.nan, np.nan, 0.0, 0.0)
                else:
                    blk = 2
                    pc = matching.match_dense(ofi[meth], gfeat[kc][meth], blk)
                    pn = matching.match_dense(ofi[meth], gfeat[kn][meth], blk)
                true_col, true_row = matching.offset_to_pixel(
                    targets[i][0] - states[off[i] + j][0],
                    targets[i][1] - states[off[i] + j][1])
                err = matching.localisation_error(pc[0], pc[1], true_col, true_row)
                recs.append({"method": meth, "dist": dist, "err": err,
                             "score": pc[2], "margin": pc[3],
                             "neg_score": pn[2], "neg_margin": pn[3]})

    tau_grid = np.arange(0.30, 0.96, 0.025)
    margin_grid = np.arange(0.0, 0.31, 0.02)
    out = {"source": "teacher_train_seen (independent of validation)",
           "n_states": len(picked), "tau": {}, "margin_tau": {}, "f1": {}, "grid": {}}
    for meth in METHODS:
        sub = [r for r in recs if r["method"] == meth]
        pos = np.array([max(r["err"], 0) <= 10.0 if np.isfinite(r["err"]) else False
                        for r in sub])
        sc = np.array([r["score"] for r in sub])
        mg = np.array([r["margin"] for r in sub])
        nsc = np.array([r["neg_score"] for r in sub])
        nmg = np.array([r["neg_margin"] for r in sub])
        best = (-1, None, None)
        for tau in tau_grid:
            for mt in margin_grid:
                acc = (sc >= tau) & (mg >= mt)
                neg_acc = (nsc >= tau) & (nmg >= mt)
                tp = int(np.sum(pos & acc))
                fn = int(np.sum(pos)) - tp
                fp = int(np.sum(~pos & acc)) + int(np.sum(neg_acc))
                f1 = 2 * tp / max(2 * tp + fp + fn, 1)
                if f1 > best[0]:
                    best = (f1, float(tau), float(mt))
        fp_neg = float(np.mean((nsc >= best[1]) & (nmg >= best[2])))
        out["tau"][meth], out["margin_tau"][meth] = best[1], best[2]
        out["f1"][meth] = best[0]
        out["grid"][meth] = {"positive_rate": float(np.mean(pos)),
                             "neg_goal_false_accept": fp_neg}
        print(f"{meth}: tau={best[1]:.3f} margin={best[2]:.2f} F1={best[0]:.3f} "
              f"pos_rate={np.mean(pos):.3f} neg_false_accept={fp_neg:.3f}")
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
