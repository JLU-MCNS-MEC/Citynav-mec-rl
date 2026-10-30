#!/usr/bin/env python3
"""Build the Stage-06 task manifest: all Val-Seen + Val-Unseen 03B tasks.

* no filtering: every 03B validation task is used, including tasks whose human
  path never entered the goal (that filter belonged to the old 03/04 pipeline
  and does not apply to the direct target-directed teacher);
* length group comes from the 03B CSV (boundaries fixed on Train-Seen:
  short <= 29 s, medium <= 49 s, long > 49 s); nothing is recomputed here;
* each (split, length_group) pool is shuffled with seed 20261030 and split into
  two disjoint cohorts, Round A / Round B, so the union covers every task;
* target-image validity is measured on the real raster and recorded.  A target
  patch with no valid elevation pixel at all is flagged goal_image_invalid and
  counted as an end-to-end failure; nothing is dropped.

Output: results/seed42/manifest.json, manifest.csv, manifest.sha256
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
sys.path.insert(0, str(ROOT / "05_transformer_m_training/scripts"))
sys.path.insert(0, str(ROOT / "04_visual_goal_global_map_pipeline/scripts"))

from dataset_v2 import RasterStore  # noqa: E402
from pipeline import TARGET_HALF_M  # noqa: E402

TEACHER = ROOT / "03b_target_directed_teacher/results"
SPLITS = ("val_seen", "val_unseen")
GROUPS = ("short", "medium", "long")
SEED = 20261030
CHECKPOINT_SHA = "077f01a4f1a643c3db7ff6a4eb57980a401232bcc4810b05925fb1fceccc0ce2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(HERE.parent / "results/seed42"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rasters = RasterStore()
    tasks: list[dict] = []
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        with (TEACHER / f"teacher_{split}.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        rng = np.random.default_rng(args.seed)
        per_group = {}
        for group in GROUPS:
            pool = [r for r in rows if r["length_group"] == group]
            order = rng.permutation(len(pool))
            a = [pool[i] for i in order[0::2]]
            b = [pool[i] for i in order[1::2]]
            per_group[group] = (a, b)
        for group, (a, b) in per_group.items():
            for cohort, chunk in (("A", a), ("B", b)):
                for r in chunk:
                    name = r["map"]
                    raster = rasters.get(name)
                    tx, ty = float(r["target_x"]), float(r["target_y"])
                    frac = raster.valid_fraction(tx, ty, TARGET_HALF_M)
                    cov = raster.coverage(tx, ty, TARGET_HALF_M)
                    center = raster.height_valid(tx, ty)
                    tasks.append({
                        "episode_id": f"{split}:{int(r['index']):05d}",
                        "split": split, "index": int(r["index"]), "map": name,
                        "length_group": group, "cohort": cohort,
                        "start_x": float(r["start_x"]), "start_y": float(r["start_y"]),
                        "start_yaw": float(r["start_yaw"]),
                        "target_x": tx, "target_y": ty,
                        "teacher_steps": int(r["steps"]),
                        "teacher_path_len_m": float(r["path_len_m"]),
                        "teacher_min_dist": float(r["min_dist"]),
                        "target_valid_fraction": float(frac),
                        "target_coverage": float(cov),
                        "target_center_valid": bool(center),
                        "goal_image_invalid": bool(frac == 0.0),
                    })
        counts[split] = {g: sum(1 for t in tasks if t["split"] == split and t["length_group"] == g)
                         for g in GROUPS}

    ids = [t["episode_id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate episode ids"
    by_split_group_cohort: dict[str, dict[str, dict[str, int]]] = {}
    for split in SPLITS:
        by_split_group_cohort[split] = {}
        for group in GROUPS:
            sel = [t for t in tasks if t["split"] == split and t["length_group"] == group]
            by_split_group_cohort[split][group] = {
                "A": sum(1 for t in sel if t["cohort"] == "A"),
                "B": sum(1 for t in sel if t["cohort"] == "B"),
                "total": len(sel)}

    manifest = {
        "seed": args.seed,
        "splits": list(SPLITS),
        "length_groups": {"short": "<= 29 steps", "medium": "30-49 steps", "long": "> 49 steps"},
        "length_group_boundaries_source": "03B, fixed on Train-Seen",
        "checkpoint_sha": CHECKPOINT_SHA,
        "candidate_library": str(ROOT / "04_visual_goal_global_map_pipeline/cache_v2/candidates24"),
        "candidate_version": "candidates24",
        "encoder": "frozen torchvision resnet50 IMAGENET1K_V2 (layer4, fp16)",
        "target_half_m": TARGET_HALF_M,
        "counts": by_split_group_cohort,
        "tasks": tasks,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    (out_dir / "manifest.sha256").write_text(h + "\n", encoding="utf-8")

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(tasks[0]))
        writer.writeheader()
        writer.writerows(tasks)
    print(f"tasks: {len(tasks)} ({counts})")
    for split in SPLITS:
        print(f"  {split}: " + ", ".join(
            f"{g}={by_split_group_cohort[split][g]['A']}+{by_split_group_cohort[split][g]['B']}"
            for g in GROUPS))
    invalid = [t["episode_id"] for t in tasks if t["goal_image_invalid"]]
    low = [t["episode_id"] for t in tasks if 0 < t["target_valid_fraction"] < 0.5]
    print(f"goal_image_invalid: {len(invalid)} {invalid[:5]}")
    print(f"target_valid_fraction < 0.5: {len(low)}")
    print(f"manifest sha256: {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
