#!/usr/bin/env python3
"""Train-Seen calibration of the continuous-belief evidence power and discount.

Runs the C2 gate along sampled teacher trajectories and grid-searches eta / rho
minimising the premature-STOP rate (estimated <=20 m while the true distance is
>25 m), then the median estimate error.  No validation split is touched.
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
from continuous_belief import ContinuousBelief, subpixel_match  # noqa: E402
from dataset_v2 import CandidateStore, FrozenEncoder, RasterStore  # noqa: E402

TEACHER = ROOT / "experiments/03b_target_directed_teacher/results"
RES = HERE.parent / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cal = json.loads((RES / "09c_calibration.json").read_text(encoding="utf-8"))
    gate = {"tau2": cal["tau2"], "margin2": cal["margin2"], "agree_m": cal["agree_m"],
            "v_min": cal["v_min"], "border_px": 12.0, "d_max": cal["d_max"]}
    sigma_vis = cal["sigma_vis_m"]
    z = np.load(TEACHER / "teacher_train_seen.npz")
    keys = [int(k) for k in z["keys"]]
    off, states, targets = z["offsets"], z["states"], z["targets"]
    with (TEACHER / "teacher_train_seen.csv").open(encoding="utf-8") as fh:
        rows = {int(r["index"]): r for r in csv.DictReader(fh)}
    rng = np.random.default_rng(20261031)
    enc = matching.Encoder(args.device)
    fe = FrozenEncoder(args.device)
    rasters, cands = RasterStore(), CandidateStore()
    episodes = []
    for i in rng.choice(len(keys), args.episodes, replace=False):
        i = int(i)
        name = rows[keys[i]]["map"]
        raster = rasters.get(name)
        lib = cands.get(name)
        t = targets[i]
        goal = matching.prepare_goal(raster, t[0], t[1])
        gf = enc.features(goal[None], ("layer2",))["layer2"][0]
        s = states[off[i]:off[i + 1]]
        obs = [matching.prepare_observation(raster, p[0], p[1], p[2]) for p in s]
        of = enc.features(np.stack(obs), ("layer2",))["layer2"]
        tvec = fe.embed([raster.north_crop(float(t[0]), float(t[1]), 48.0)])[0].astype(np.float32)
        sims = (lib.embeddings @ tvec).astype(np.float64)
        recs = []
        for j in range(len(s)):
            m = subpixel_match(of[j], gf, 2)
            dx, dy = matching.pixel_to_offset(m["col"], m["row"])
            zz = np.array([s[j, 0] + dx, s[j, 1] + dy])
            recs.append({"pos": s[j, :2].astype(np.float64), "z": zz,
                         "score": m["score"], "margin": m["margin"],
                         "validity": float(raster.valid_fraction(float(zz[0]), float(zz[1]), 8.0)),
                         "border": min(m["col"], m["row"], 223 - m["col"], 223 - m["row"]),
                         "true_dist": float(np.linalg.norm(s[j, :2] - t))})
        episodes.append({"positions": lib.positions.astype(np.float64), "sims": sims,
                         "target": t.astype(np.float64), "states": recs})

    def evaluate(eta, rho, T=0.1):
        errs, ents = [], []
        prem = good = 0
        for ep in episodes:
            b = ContinuousBelief(ep["positions"], ep["sims"], T, k=8, sigma_prior_m=12.0)
            last = -99
            for j, st in enumerate(ep["states"]):
                agree = b.last_accept is not None and float(
                    np.linalg.norm(st["z"] - b.last_accept)) <= gate["agree_m"]
                maha = b.best_mahalanobis(st["z"])
                conf_ok = (st["score"] >= gate["tau2"] and st["margin"] >= gate["margin2"]
                           and st["validity"] >= gate["v_min"]
                           and st["border"] >= gate["border_px"] and maha <= gate["d_max"])
                if conf_ok:
                    b.last_accept = st["z"].copy()
                if conf_ok and agree:
                    b.update(st["z"], sigma_vis, eta, rho, j)
                est = b.estimate()
                d_est = float(np.linalg.norm(est - st["pos"]))
                d_true = st["true_dist"]
                if d_est <= 20:
                    if d_true > 25:
                        prem += 1
                    else:
                        good += 1
                errs.append(float(np.linalg.norm(est - ep["target"])))
                ents.append(b.entropy())
        return (float(np.median(errs)), float(np.mean(ents)),
                prem / max(prem + good, 1), int(good), int(prem))

    grid = []
    best = None
    for eta in (0.1, 0.3, 0.6):
        for rho in (0.5, 0.8):
            med, ent, prec, good, bad = evaluate(eta, rho)
            row = {"eta": eta, "rho": rho, "median_est_err_m": med,
                   "mean_entropy": ent, "premature_stop_rate": prec,
                   "stops_good": good, "stops_premature": bad}
            grid.append(row)
            if ent < 0.10:
                continue
            key = (prec, med)
            if best is None or key < (best["premature_stop_rate"], best["median_est_err_m"]):
                best = row
    assert best is not None
    out = {"source": "100 Train-Seen teacher episodes (seed 20261031)",
           "sigma_vis_m": sigma_vis, "gate": gate, "grid": grid, "best": best}
    (RES / "09c_belief_calibration.json").write_text(json.dumps(out, indent=2) + "\n",
                                                     encoding="utf-8")
    print(json.dumps(best, indent=1))
    for g in grid:
        print(g)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
