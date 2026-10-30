#!/usr/bin/env python3
"""Build the Test-Unseen task manifest from the official split (5,311 tasks).

Start pose and target are parsed with the same helper the 03B teacher used
(``direct_teacher.start_pose``), so the test tasks are defined identically to
the validation tasks.  No teacher labels are generated for this split, so
``length_group`` is null; the report stratifies by map and by start distance.

Cohorts A/B are assigned with the same seed (20261030) used for the validation
manifest, alternated after a deterministic shuffle of the whole split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "03b_target_directed_teacher/scripts"))
sys.path.insert(0, str(ROOT / "04_visual_goal_global_map_pipeline/scripts"))
sys.path.insert(0, str(ROOT / "05_transformer_m_training/scripts"))

from dataset_v2 import RasterStore  # noqa: E402
from direct_teacher import start_pose  # noqa: E402
from pipeline import TARGET_HALF_M  # noqa: E402

SPLIT_JSON = Path("/home/20261024_1/Workspace/DATA/sbfnav-data/original_citynav/citynav_test_unseen.json")
CHECKPOINT_SHA = "077f01a4f1a643c3db7ff6a4eb57980a401232bcc4810b05925fb1fceccc0ce2"
SEED = 20261030


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(HERE.parent / "results/test_unseen"))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    rasters = RasterStore()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(records))
    cohort = {int(i): ("A" if pos % 2 == 0 else "B") for pos, i in enumerate(order)}
    tasks = []
    for i, rec in enumerate(records):
        xy, yaw = start_pose(rec)
        target = np.asarray(rec["target_positions"][-1], np.float64)[:2]
        name = f"{rec['area']}_block_{rec['block']}"
        raster = rasters.get(name)
        tx, ty = float(target[0]), float(target[1])
        frac = raster.valid_fraction(tx, ty, TARGET_HALF_M)
        tasks.append({
            "episode_id": f"test_unseen:{i:05d}", "split": "test_unseen", "index": i,
            "map": name, "length_group": None, "cohort": cohort[i],
            "start_x": float(xy[0]), "start_y": float(xy[1]), "start_yaw": float(yaw),
            "target_x": tx, "target_y": ty,
            "teacher_steps": None, "teacher_path_len_m": None, "teacher_min_dist": None,
            "target_valid_fraction": float(frac),
            "target_coverage": float(raster.coverage(tx, ty, TARGET_HALF_M)),
            "target_center_valid": bool(raster.height_valid(tx, ty)),
            "goal_image_invalid": bool(frac == 0.0),
        })
    ids = [t["episode_id"] for t in tasks]
    assert len(ids) == len(set(ids)) == 5311
    manifest = {
        "seed": SEED, "splits": ["test_unseen"],
        "length_groups": None,
        "length_group_boundaries_source": "not applicable: no 03B teacher labels for test_unseen",
        "checkpoint_sha": CHECKPOINT_SHA,
        "candidate_library": str(ROOT / "04_visual_goal_global_map_pipeline/cache_v2/candidates24"),
        "candidate_version": "candidates24",
        "encoder": "frozen torchvision resnet50 IMAGENET1K_V2 (layer4, fp16)",
        "target_half_m": TARGET_HALF_M,
        "counts": {"test_unseen": {"A": sum(1 for t in tasks if t["cohort"] == "A"),
                                   "B": sum(1 for t in tasks if t["cohort"] == "B"),
                                   "total": len(tasks)}},
        "tasks": tasks,
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "manifest.sha256").write_text(sha + "\n", encoding="utf-8")
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(tasks[0]))
        writer.writeheader()
        writer.writerows(tasks)
    maps = {}
    for t in tasks:
        maps[t["map"]] = maps.get(t["map"], 0) + 1
    print(f"tasks: {len(tasks)} | cohorts A={manifest['counts']['test_unseen']['A']} "
          f"B={manifest['counts']['test_unseen']['B']}")
    print("maps:", maps)
    print(f"goal_image_invalid: {sum(1 for t in tasks if t['goal_image_invalid'])}; "
          f"valid<0.5: {sum(1 for t in tasks if 0 < t['target_valid_fraction'] < 0.5)}")
    print(f"manifest sha256: {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
