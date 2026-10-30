#!/usr/bin/env python3
"""Calibrate the dynamic-belief parameters on Train-Seen only.

Runs the visual matcher along a fixed sample of Train-Seen teacher trajectories
and grid-searches the prior temperature, visual variance, evidence power and
correlated-frame discount.  The selection criterion is the median MAP-candidate
distance to the true target, with an entropy floor so the posterior cannot
collapse to a single mode.

Writes results/09b_calibration.json (frozen for the closed-loop run).
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
from dataset_v2 import CandidateStore, FrozenEncoder, RasterStore  # noqa: E402

TEACHER = ROOT / "03b_target_directed_teacher/results"
RES = HERE.parent / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    calib = json.loads((RES / "09a_calibration.json").read_text(encoding="utf-8"))
    tau = calib["tau"]["layer2"]
    mtau = calib["margin_tau"]["layer2"]

    z = np.load(TEACHER / "teacher_train_seen.npz")
    keys = [int(k) for k in z["keys"]]
    off = z["offsets"]
    states, targets = z["states"], z["targets"]
    with (TEACHER / "teacher_train_seen.csv").open(encoding="utf-8") as fh:
        rows = {int(r["index"]): r for r in csv.DictReader(fh)}
    rng = np.random.default_rng(20261030)
    pick = rng.choice(len(keys), size=min(args.episodes, len(keys)), replace=False)

    encoder = matching.Encoder(args.device)
    stage05_encoder = FrozenEncoder(args.device)
    rasters, cands = RasterStore(), CandidateStore()
    episodes = []
    for i in sorted(pick):
        key = keys[i]
        name = rows[key]["map"]
        s = states[off[i]:off[i + 1]]
        t = targets[i]
        lib = cands.get(name)
        raster = rasters.get(name)
        goal = matching.prepare_goal(raster, t[0], t[1])
        gfeat = encoder.features(goal[None], ("layer2",))["layer2"][0]
        obs = [matching.prepare_observation(raster, p[0], p[1], p[2]) for p in s]
        of = encoder.features(np.stack(obs), ("layer2",))["layer2"]
        # prior uses the same target embedding as Stage 06 (Stage-05 encoder)
        tvec = stage05_encoder.embed(
            [raster.north_crop(float(t[0]), float(t[1]), 48.0)])[0].astype(np.float32)
        sims = lib.embeddings @ tvec
        per_state = []
        for j in range(len(s)):
            m = matching.match_dense(of[j], gfeat, 2)
            col, row = m[0], m[1]
            dx, dy = matching.pixel_to_offset(col, row)
            conf = float(m[2] >= tau and m[3] >= mtau)
            d_true = float(np.linalg.norm(s[j, :2] - t))
            col_t, row_t = matching.offset_to_pixel(t[0] - s[j, 0], t[1] - s[j, 1])
            in_view = (d_true <= 48.0 and 0 <= col_t <= 223 and 0 <= row_t <= 223)
            per_state.append({"pos": s[j, :2].astype(np.float64),
                              "r_vis": np.array([s[j, 0] + dx, s[j, 1] + dy]),
                              "conf": conf and in_view, "score": float(m[2]),
                              "in_view": bool(in_view)})
        episodes.append({"name": name, "target": t.astype(np.float64),
                         "positions": lib.positions.astype(np.float64),
                         "sims": sims.astype(np.float64), "states": per_state})

    def evaluate(T, sigma, eta, rho, agree_m=0.0):
        errs, ents = [], []
        premature = good_stop = 0
        for ep in episodes:
            logits = ep["sims"] / T
            logp = logits - logits.max()
            logp -= np.log(np.exp(logp).sum())
            last = -99
            last_conf = None
            for j, st in enumerate(ep["states"]):
                do_update = bool(st["conf"])
                if do_update and agree_m > 0:
                    ok = last_conf is not None and float(
                        np.linalg.norm(st["r_vis"] - last_conf)) <= agree_m
                    last_conf = st["r_vis"]
                    do_update = ok
                if do_update:
                    d = ep["positions"] - st["r_vis"]
                    d2 = np.einsum("ij,ij->i", d, d)
                    ll = -0.5 * d2 / (sigma * sigma)
                    w = eta * (rho if j - last <= 1 else 1.0)
                    logp = logp + w * (ll - ll.max())
                    logp -= np.log(np.exp(logp).sum())
                    last = j
                k = int(np.argmax(logp))
                err = float(np.linalg.norm(ep["positions"][k] - ep["target"]))
                true_d = float(np.linalg.norm(st["pos"] - ep["target"]))
                est_d = float(np.linalg.norm(ep["positions"][k] - st["pos"]))
                if est_d <= 20.0:
                    if true_d > 25.0:
                        premature += 1
                    else:
                        good_stop += 1
                if st["in_view"]:
                    errs.append(err)
                    p = np.exp(logp)
                    ents.append(float(-(p * np.log(p + 1e-12)).sum() / math.log(len(p))))
        prec = premature / max(premature + good_stop, 1)
        return (float(np.median(errs)), float(np.mean(ents)), float(prec),
                int(good_stop), int(premature))

    best = None
    grid = []
    for T in (0.05, 0.1, 0.2):
        for sigma in (5.0, 10.0):
            for eta in (0.1, 0.3):
                for agree in (0.0, 10.0, 20.0):
                    med, ent, prec, good, bad = evaluate(T, sigma, eta, 0.7, agree)
                    row = {"T": T, "sigma": sigma, "eta": eta, "rho": 0.7,
                           "agree_m": agree, "median_map_err_m": med,
                           "mean_entropy": ent, "premature_stop_rate": prec,
                           "stops_good": good, "stops_premature": bad}
                    grid.append(row)
                    if ent < 0.15:
                        continue
                    key = (prec, med)
                    if best is None or key < (best["premature_stop_rate"],
                                              best["median_map_err_m"]):
                        best = row
    assert best is not None
    gate_grid = [g for g in grid if g["T"] == best["T"] and g["sigma"] == best["sigma"]
                 and g["eta"] == best["eta"]]
    out = {"source": "150 Train-Seen teacher episodes (seed 20261030)",
           "rule": "min premature-stop rate, then min median MAP error, entropy >= 0.15",
           "gate_grid": gate_grid,
           "episodes": len(episodes), "best": best,
           "rule": "min median MAP distance; mean normalised entropy >= 0.15",
           "grid": grid, "frozen": True}
    (RES / "09b_calibration.json").write_text(json.dumps(out, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(best, indent=1))
    for g in sorted(grid, key=lambda x: x["median_map_err_m"])[:6]:
        print(g)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
