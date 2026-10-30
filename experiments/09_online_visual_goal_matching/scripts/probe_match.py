#!/usr/bin/env python3
"""Quick validation + speed probe for the three matchers on real states."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
import sys
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "04_visual_goal_global_map_pipeline/scripts"))
sys.path.insert(0, str(ROOT / "05_transformer_m_training/scripts"))

import geom  # noqa: E402
import matching  # noqa: E402
from dataset_v2 import RasterStore  # noqa: E402

STAGE06 = ROOT / "06_closed_loop_success_rate/results/seed42"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--max-dist", type=float, default=45.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = {t["episode_id"]: t for t in
                json.loads((STAGE06 / "manifest.json").read_text(encoding="utf-8"))["tasks"]}
    files = sorted((STAGE06 / "episodes").glob("*.json"))
    rasters = RasterStore()
    states = []
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        t = manifest[rec["episode_id"]]
        for s in rec["trajectory"][:: max(1, len(rec["trajectory"]) // 3)]:
            d = math.hypot(t["target_x"] - s["x"], t["target_y"] - s["y"])
            if d <= args.max_dist:
                states.append((rec["episode_id"], rec["map"], s, t))
                break
        if len(states) >= args.n:
            break
    print(f"states: {len(states)} (target within {args.max_dist} m)")
    encoder = matching.Encoder(args.device)
    obs_imgs, goals, metas = [], [], []
    t0 = time.time()
    for eid, name, s, t in states:
        raster = rasters.get(name)
        obs = matching.prepare_observation(raster, s["x"], s["y"], s["yaw"])
        goal = matching.prepare_goal(raster, t["target_x"], t["target_y"])
        true_col, true_row = matching.offset_to_pixel(t["target_x"] - s["x"],
                                                      t["target_y"] - s["y"])
        obs_imgs.append(obs)
        goals.append(goal)
        metas.append((eid, s, t, true_col, true_row))
    t_crop = time.time() - t0
    t1 = time.time()
    feats = encoder.features(np.stack(obs_imgs), ("layer2", "layer3"))
    gfeats = encoder.features(np.stack(goals), ("layer2", "layer3"))
    t_enc = time.time() - t1
    print(f"crop {t_crop/len(states)*1000:.2f} ms/state | encode "
          f"{t_enc/len(states)*1000:.2f} ms/state (both images, 2 layers)")
    errs = {"rgb": [], "layer3": [], "layer2": []}
    for i, (eid, s, t, tc, tr) in enumerate(metas):
        r = matching.match_rgb(obs_imgs[i], goals[i])
        e = matching.localisation_error(r[0], r[1], tc, tr)
        errs["rgb"].append(e)
        for layer, block in (("layer3", 2), ("layer2", 2)):
            m = matching.match_dense(feats[layer][i], gfeats[layer][i], block)
            errs[layer].append(matching.localisation_error(m[0], m[1], tc, tr))
        if i < 8:
            print(f"{eid} d={math.dist([s['x'],s['y']],[t['target_x'],t['target_y']]):5.1f}m "
                  f"rgb={errs['rgb'][-1]:6.1f}m L3={errs['layer3'][-1]:6.1f}m "
                  f"L2={errs['layer2'][-1]:6.1f}m")
    for k, v in errs.items():
        print(f"{k}: median {np.median(v):5.1f} m | mean {np.mean(v):5.1f} m | "
              f"<20m {np.mean(np.array(v) < 20):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
