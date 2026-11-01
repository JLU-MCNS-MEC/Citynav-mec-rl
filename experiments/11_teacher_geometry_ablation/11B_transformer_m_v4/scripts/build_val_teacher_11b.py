#!/usr/bin/env python3
"""11B: build the Stage-03-style four-action teacher on Val-Seen.

VALIDATION ONLY -- these labels are used exclusively for the fixed validation
metric (best.pt selection + early stop), never as training data (doc 2.1).

Uses the exact Stage 03 conversion code (four_action_teacher.convert_episode)
on the original citynav_val_seen.json, so validation supervision matches the
training supervision of M-v4 / M-v5.

Output (results/cache/val_teacher/):
  teacher_val_seen.npz   keys / offsets / states / actions / targets
  teacher_val_seen.csv   index, map, success, steps
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
S03 = REPO / "experiments/03_fixed50m_four_action_teacher/scripts"
sys.path.insert(0, str(S03))
from four_action_teacher import convert_episode  # noqa: E402

DATA = Path("/mnt/hdd2t/Citynav-MEC/DATA/sbfnav-data/original_citynav")
GROUND_SRC = Path("/tmp/citynav-official/gsamllavanav/mapdata.py")
OUT = HERE.parent / "results" / "cache" / "val_teacher"


def load_ground() -> dict[str, float]:
    if not GROUND_SRC.exists():
        return {}
    import importlib.util
    spec = importlib.util.spec_from_file_location("official_mapdata", GROUND_SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: float(v) for k, v in module.GROUND_LEVEL.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val_seen")
    args = parser.parse_args()
    import json
    records = json.loads((DATA / f"citynav_{args.split}.json").read_text(encoding="utf-8"))
    ground = load_ground()
    OUT.mkdir(parents=True, exist_ok=True)

    keys, states, actions, targets, rows = [], [], [], [], []
    for idx, record in enumerate(records):
        name = f"{record['area']}_block_{record['block']}"
        result = convert_episode(record, ground.get(name))
        if not result.get("success", False) or not result.get("states"):
            continue
        keys.append(f"{idx}")
        states.append(np.asarray(result["states"], np.float32))
        actions.append(np.asarray(result["actions"], np.int8))
        targets.append(np.asarray(result["target_xy"], np.float32))
        rows.append({"index": idx, "map": name, "success": True,
                     "steps": len(result["states"])})

    offsets = np.concatenate([[0], np.cumsum([len(a) for a in actions])]).astype(np.int64)
    np.savez_compressed(
        OUT / f"teacher_{args.split}.npz",
        keys=np.asarray(keys),
        offsets=offsets,
        states=np.concatenate(states),
        actions=np.concatenate(actions),
        targets=np.stack(targets),
    )
    with (OUT / f"teacher_{args.split}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["index", "map", "success", "steps"])
        w.writeheader()
        w.writerows(rows)
    print(f"{args.split}: {len(rows)}/{len(records)} episodes converted, "
          f"{int(offsets[-1])} states", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
