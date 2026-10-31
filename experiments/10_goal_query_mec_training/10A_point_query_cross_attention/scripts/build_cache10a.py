#!/usr/bin/env python3
"""10A training image cache (Train-Seen teacher states, resumable).

Deterministic state sampling (no RNG): teacher trajectories in npz order,
09A round-robin distance-bin stratification, up to 12 states per episode.
The 2,000-state calibration subset (calibrate_09a.py reproduction) is
excluded so tau calibration never touches training data.

Cache layout (results/cache/):
  manifest.csv          state_id, ep, j, map, x, y, yaw, target_x, target_y,
                        true_col, true_row, in_view, dist_m, dist_bin,
                        length_group, obs_file, goal_file
  imgs/obs_{k:07d}.npy  224x224x3 uint8 north-up observation
  goals/goal_{h}.npy    224x224x3 uint8 north-up goal (dedup by goal key)
  meta.json             counts + per-bin stats + source SHA

Resume: a state is done iff its obs file exists; manifest rows are appended
per chunk and done states are skipped on restart.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common10a
from common10a import (BIN_LABELS, RES, bin_of, calibration_subset, geom,
                       in_view_pixel, load_teacher, matching, pick_steps,
                       require_data_root)

CACHE = RES / "cache"
IMGS = CACHE / "imgs"
GOALS = CACHE / "goals"


def main() -> int:
    require_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk", type=int, default=512)
    args = parser.parse_args()

    from dataset_v2 import RasterStore  # local import: heavy deps after arg parse

    rows, keys, off, states, targets = load_teacher()
    calib = {(i, j) for i, j in calibration_subset()}

    CACHE.mkdir(parents=True, exist_ok=True)
    IMGS.mkdir(parents=True, exist_ok=True)
    GOALS.mkdir(parents=True, exist_ok=True)

    # ---- plan the deterministic state list ---------------------------------
    plan: list[dict] = []
    per_bin: dict[str, int] = {l: 0 for l in BIN_LABELS}
    per_bin_inview: dict[str, int] = {l: 0 for l in BIN_LABELS}
    center = geom.center_px()
    for i, key in enumerate(keys):
        s = states[off[i]:off[i + 1]]
        t = targets[i]
        dists = np.linalg.norm(s[:, :2] - t, axis=1)
        picked = pick_steps([(j, float(dists[j])) for j in range(len(s))])
        for j in picked:
            if (i, j) in calib:
                continue
            dx, dy = t[0] - s[j][0], t[1] - s[j][1]
            true_col = center + dx / geom.CUR_SCALE
            true_row = center - dy / geom.CUR_SCALE
            hc, hr = geom.heading_pixel(s[j][0], s[j][1], s[j][2], t[0], t[1])
            inview = in_view_pixel(hc, hr)
            dist = float(dists[j])
            label = BIN_LABELS[bin_of(dist)]
            plan.append({"ep": i, "j": int(j), "map": rows[key]["map"],
                         "x": float(s[j][0]), "y": float(s[j][1]),
                         "yaw": float(s[j][2]), "target_x": float(t[0]),
                         "target_y": float(t[1]), "true_col": true_col,
                         "true_row": true_row, "in_view": inview,
                         "dist_m": dist, "dist_bin": label,
                         "length_group": rows[key]["length_group"],
                         "goal_key": (rows[key]["map"], round(float(t[0]), 2),
                                      round(float(t[1]), 2))})
            per_bin[label] += 1
            if inview:
                per_bin_inview[label] += 1

    # ---- goal dedup registry -------------------------------------------------
    goal_by_key: dict[tuple, str] = {}
    for p in plan:
        gk = p["goal_key"]
        if gk not in goal_by_key:
            h = hashlib.sha256(f"{gk[0]}|{gk[1]}|{gk[2]}".encode()).hexdigest()[:12]
            goal_by_key[gk] = f"goal_{h}.npy"

    # ---- resume bookkeeping ---------------------------------------------------
    manifest_path = CACHE / "manifest.csv"
    fieldnames = ["state_id", "ep", "j", "map", "x", "y", "yaw", "target_x",
                  "target_y", "true_col", "true_row", "in_view", "dist_m",
                  "dist_bin", "length_group", "obs_file", "goal_file"]
    done_ids: set[str] = set()
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done_ids.add(r["state_id"])
    pending = [(f"obs_{k:07d}", p) for k, p in enumerate(plan)
               if f"obs_{k:07d}" not in done_ids]
    print(f"plan states {len(plan)} | done {len(done_ids)} | pending {len(pending)} "
          f"| unique goals {len(goal_by_key)}", flush=True)

    # ---- build ------------------------------------------------------------------
    rasters = RasterStore()
    start = time.time()
    n = 0
    with manifest_path.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if not done_ids:
            w.writeheader()
        for i0 in range(0, len(pending), args.chunk):
            chunk = pending[i0:i0 + args.chunk]
            for sid, p in chunk:
                raster = rasters.get(p["map"])
                gf = goal_by_key[p["goal_key"]]
                gpath = GOALS / gf
                if not gpath.exists():
                    gimg = matching.prepare_goal(raster, p["target_x"], p["target_y"])
                    np.save(gpath, gimg)
                oimg = matching.prepare_observation(raster, p["x"], p["y"], p["yaw"])
                np.save(IMGS / f"{sid}.npy", oimg)
                row = {"state_id": sid, "ep": p["ep"], "j": p["j"], "map": p["map"],
                       "x": round(p["x"], 4), "y": round(p["y"], 4),
                       "yaw": round(p["yaw"], 6), "target_x": p["target_x"],
                       "target_y": p["target_y"], "true_col": round(p["true_col"], 2),
                       "true_row": round(p["true_row"], 2), "in_view": int(p["in_view"]),
                       "dist_m": round(p["dist_m"], 3), "dist_bin": p["dist_bin"],
                       "length_group": p["length_group"], "obs_file": f"imgs/{sid}.npy",
                       "goal_file": f"goals/{gf}"}
                w.writerow(row)
                n += 1
            fh.flush()
            if (i0 // args.chunk) % 10 == 0:
                rate = n / max(time.time() - start, 1e-9)
                print(f"{n}/{len(pending)} | {rate:.1f} st/s | elapsed "
                      f"{time.time() - start:.0f}s | eta "
                      f"{(len(pending) - n) / rate:.0f}s", flush=True)

    meta = {"states": len(plan), "calibration_excluded": len(calib),
            "per_bin": per_bin, "per_bin_in_view": per_bin_inview,
            "unique_goals": len(goal_by_key),
            "seconds": round(time.time() - start, 1),
            "source_npz_sha": hashlib.sha256(
                common10a.TEACHER_NPZ.read_bytes()).hexdigest(),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (CACHE / "meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                     encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
