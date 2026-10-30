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
            per_state.append({"pos": s[j, :2].astype(np.float64),
                              "r_vis": np.array([s[j, 0] + dx, s[j, 1] + dy]),
                              "conf": conf, "score": float(m[2])})
        episodes.append({"name": name, "target": t.astype(np.float64),
                         "positions": lib.positions.astype(np.float64),
                         "sims": sims.astype(np.float64), "states": per_state})

    def evaluate(T, sigma, eta, rho):
        errs, ents = [], []
        for ep in episodes:
            logits = ep["sims"] / T
            logp = logits - logits.max()
            logp -= np.log(np.exp(logp).sum())
            last = -99
            for j, st in enumerate(ep["states"]):
                if st["conf"]:
                    d = ep["positions"] - st["r_vis"]
                    d2 = np.einsum("ij,ij->i", d, d)
                    ll = -0.5 * d2 / (sigma * sigma)
                    w = eta * (rho if j - last <= 1 else 1.0)
                    logp = logp + w * (ll - ll.max())
                    logp -= np.log(np.exp(logp).sum())
                    last = j
                k = int(np.argmax(logp))
                errs.append(float(np.linalg.norm(ep["positions"][k] - ep["target"])))
                p = np.exp(logp)
                ents.append(float(-(p * np.log(p + 1e-12)).sum() / math.log(len(p))))
        return float(np.median(errs)), float(np.mean(ents))

    best = None
    grid = []
    for T in (0.05, 0.1, 0.2):
        for sigma in (5.0, 10.0, 20.0):
            for eta in (0.1, 0.3, 0.6):
                for rho in (0.3, 0.7):
                    med, ent = evaluate(T, sigma, eta, rho)
                    grid.append({"T": T, "sigma": sigma, "eta": eta, "rho": rho,
                                 "median_map_err_m": med, "mean_entropy": ent})
                    if ent < 0.15:          # collapsed posterior -> reject
                        continue
                    if best is None or med < best["median_map_err_m"]:
                        best = {"T": T, "sigma": sigma, "eta": eta, "rho": rho,
                                "median_map_err_m": med, "mean_entropy": ent}
    assert best is not None
    out = {"source": "150 Train-Seen teacher episodes (seed 20261030)",
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
