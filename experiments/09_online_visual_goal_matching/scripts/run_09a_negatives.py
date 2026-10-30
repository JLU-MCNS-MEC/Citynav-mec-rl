#!/usr/bin/env python3
"""09A negative controls: wrong goal images against the same observations.

Every 10th sampled state is matched against four wrong goals:

1. same map, 30-80 m from the true target (hard negative, similar buildings);
2. same map, >150 m away;
3. another map;
4. the correct goal image with 16x16 pixel blocks shuffled.

Reports, per method and negative type, the score distribution and the share of
wrong goals accepted at the calibrated thresholds.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import zlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "04_visual_goal_global_map_pipeline/scripts"))
sys.path.insert(0, str(ROOT / "05_transformer_m_training/scripts"))

import matching  # noqa: E402
from dataset_v2 import RasterStore  # noqa: E402

RES = HERE.parent / "results"
STAGE06 = ROOT / "06_closed_loop_success_rate/results/seed42"
METHODS = ("rgb_ncc", "layer3", "layer2")


def shuffled(img: np.ndarray, seed: int, block: int = 16) -> np.ndarray:
    n = img.shape[0] // block
    blocks = img.reshape(n, block, n, block, 3).transpose(0, 2, 1, 3, 4)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n * n)
    out = blocks.reshape(n * n, block, block, 3)[order].reshape(n, n, block, block, 3)
    return out.transpose(0, 2, 1, 3, 4).reshape(img.shape[0], img.shape[1], 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    calib = json.loads((RES / "09a_calibration.json").read_text(encoding="utf-8"))
    tau, mt = calib["tau"], calib["margin_tau"]

    with open(RES / "09a_match_manifest.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))[::args.stride]
    for r in rows:
        for k in ("x", "y", "yaw", "target_x", "target_y", "true_dist_m"):
            r[k] = float(r[k])
    targets_by_map = defaultdict(list)
    tasks = json.loads((STAGE06 / "manifest.json").read_text(encoding="utf-8"))["tasks"]
    for t in tasks:
        targets_by_map[t["map"]].append((t["target_x"], t["target_y"]))
    for m in targets_by_map:
        targets_by_map[m] = list(dict.fromkeys(targets_by_map[m]))
    maps = sorted(targets_by_map)
    rng = np.random.default_rng(20261030)
    print(f"states: {len(rows)} | tau {tau}")

    encoder = matching.Encoder(args.device)
    rasters = RasterStore()
    img_cache: dict[tuple, np.ndarray] = {}
    feat_cache: dict[tuple, dict] = {}

    def goal(name, tx, ty):
        k = (name, round(float(tx), 2), round(float(ty), 2))
        if k not in img_cache:
            img_cache[k] = matching.prepare_goal(rasters.get(name), tx, ty)
        return k

    def ensure(keys):
        todo = [k for k in keys if k not in feat_cache]
        if todo:
            out = encoder.features(np.stack([img_cache[k] for k in todo]),
                                   ("layer2", "layer3"))
            for i, k in enumerate(todo):
                feat_cache[k] = {"layer2": out["layer2"][i], "layer3": out["layer3"][i]}

    per_state = []
    for i0 in range(0, len(rows), args.batch):
        chunk = rows[i0:i0 + args.batch]
        obs, plan = [], []
        for r in chunk:
            obs.append(matching.prepare_observation(rasters.get(r["map"]),
                                                    r["x"], r["y"], r["yaw"]))
            tx, ty = r["target_x"], r["target_y"]
            same = [p for p in targets_by_map[r["map"]]
                    if 30.0 < float(np.hypot(p[0] - tx, p[1] - ty)) <= 80.0]
            far = [p for p in targets_by_map[r["map"]]
                   if float(np.hypot(p[0] - tx, p[1] - ty)) > 150.0]
            other = [m for m in maps if m != r["map"]]
            wrong_near = same[int(rng.integers(len(same)))] if same else None
            wrong_far = far[int(rng.integers(len(far)))] if far else None
            wrong_cross = None
            if other:
                m2 = other[int(rng.integers(len(other)))]
                pool = targets_by_map[m2]
                wrong_cross = (m2, *pool[int(rng.integers(len(pool)))])
            kc = goal(r["map"], tx, ty)
            ks = goal(r["map"], *wrong_near) if wrong_near else None
            kf = goal(r["map"], *wrong_far) if wrong_far else None
            kx = goal(wrong_cross[0], wrong_cross[1], wrong_cross[2]) if wrong_cross else None
            plan.append((r, kc, ks, kf, kx))
        of = encoder.features(np.stack(obs), ("layer2", "layer3"))
        keys = []
        for p in plan:
            keys += [k for k in p[1:] if k is not None]
        ensure(keys)
        # shuffled goals are separate images
        for n, (r, kc, ks, kf, kx) in enumerate(plan):
            key = (kc, "shuf")
            if key not in img_cache:
                img_cache[key] = shuffled(img_cache[kc], seed=zlib.crc32(r["episode_id"].encode()))
            ensure([key])
            ofi = {lay: of[lay][n] for lay in ("layer2", "layer3")}
            for meth in METHODS:
                for ntype, k in (("correct", kc), ("same_map_near", ks),
                                 ("same_map_far", kf), ("cross_map", kx),
                                 ("shuffled", key)):
                    if k is None:
                        continue
                    if meth == "rgb_ncc":
                        m = matching.match_rgb(obs[n], img_cache[k]) or (0, 0, 0.0, 0.0)
                    else:
                        m = matching.match_dense(ofi[meth], feat_cache[k][meth], 2)
                    per_state.append({"method": meth, "type": ntype,
                                      "episode_id": r["episode_id"],
                                      "map": r["map"], "distance_bin": r["distance_bin"],
                                      "score": m[2], "margin": m[3],
                                      "accepted": int(m[2] >= tau[meth] and m[3] >= mt[meth])})
        print(f"{min(i0 + len(chunk), len(rows))}/{len(rows)} states", flush=True)

    with gzip.open(RES / "09a_negative_per_state.csv.gz", "wt", newline="",
                   encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_state[0]))
        w.writeheader()
        w.writerows(per_state)

    agg = defaultdict(list)
    for r in per_state:
        agg[(r["type"], r["method"])].append(r)
    out = []
    for (ntype, meth), rows_ in sorted(agg.items()):
        sc = np.array([r["score"] for r in rows_])
        ac = np.array([r["accepted"] for r in rows_])
        out.append({"negative_type": ntype, "method": meth, "n": len(rows_),
                    "score_mean": float(sc.mean()), "score_median": float(np.median(sc)),
                    "score_p90": float(np.percentile(sc, 90)),
                    "accept_rate": float(ac.mean()),
                    "high_conf_rate": float(np.mean(sc >= 0.95))})
    with (RES / "09a_negative_controls.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"{r['negative_type']:14s} {r['method']:8s} n={r['n']:5d} "
              f"score med {r['score_median']:.3f} | accept {r['accept_rate']:.3f} | "
              f">0.95 {r['high_conf_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
