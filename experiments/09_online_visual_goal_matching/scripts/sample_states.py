#!/usr/bin/env python3
"""Deterministic state sample from the Stage-06 trajectories (09A).

Up to 12 states per episode, spread over the true target-distance bins
[0,20) [20,40) [40,60) [60,80) [80,inf).  States come from the stored closed-loop
trajectories; nothing is fabricated.  The true target is used only to build the
goal image and to score the matcher offline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "04_visual_goal_global_map_pipeline/scripts"))
sys.path.insert(0, str(ROOT / "05_transformer_m_training/scripts"))
import geom  # noqa: E402
from dataset_v2 import RasterStore  # noqa: E402

STAGE06 = ROOT / "06_closed_loop_success_rate/results/seed42"
BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, float("inf"))]
LABELS = ["0-20", "20-40", "40-60", "60-80", ">80"]
PER_EPISODE = 12


def bin_of(d: float) -> int:
    for i, (lo, hi) in enumerate(BINS):
        if lo <= d < hi:
            return i
    return len(BINS) - 1


def pick_steps(steps: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Round-robin over bins, evenly spaced inside each bin, at most 12."""
    by_bin: dict[int, list[tuple[int, float]]] = {}
    for step, d in steps:
        by_bin.setdefault(bin_of(d), []).append((step, d))
    for b in by_bin:
        by_bin[b].sort()
    quota = max(1, PER_EPISODE // max(len(by_bin), 1))
    chosen: list[tuple[int, float]] = []
    for b in sorted(by_bin):
        entries = by_bin[b]
        if len(entries) <= quota:
            chosen.extend(entries)
        else:
            idx = sorted({round(i * (len(entries) - 1) / (quota - 1)) if quota > 1 else 0
                          for i in range(quota)})
            chosen.extend(entries[i] for i in idx)
    return sorted(chosen)[:PER_EPISODE]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(HERE.parent / "results/09a_match_manifest.csv"))
    parser.add_argument("--meta", default=str(HERE.parent / "results/09a_match_manifest.json"))
    args = parser.parse_args()
    manifest = {t["episode_id"]: t for t in
                json.loads((STAGE06 / "manifest.json").read_text(encoding="utf-8"))["tasks"]}
    rasters = RasterStore()
    rows = []
    for path in sorted((STAGE06 / "episodes").glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        task = manifest[rec["episode_id"]]
        steps = [(s["step"], math.hypot(task["target_x"] - s["x"],
                                        task["target_y"] - s["y"]))
                 for s in rec["trajectory"]]
        if not steps:
            continue
        raster = rasters.get(rec["map"])
        for step, dist in pick_steps(steps):
            s = rec["trajectory"][step]
            col, row = geom.heading_pixel(s["x"], s["y"], s["yaw"],
                                          task["target_x"], task["target_y"])
            in_view = bool(0 <= col <= geom.OUT - 1 and 0 <= row <= geom.OUT - 1)
            _, valid = raster.heading_crop(float(s["x"]), float(s["y"]), float(s["yaw"]),
                                           return_valid=True)
            rows.append({
                "episode_id": rec["episode_id"], "split": rec["split"],
                "map": rec["map"], "length_group": rec["length_group"],
                "step": step, "x": round(s["x"], 4), "y": round(s["y"], 4),
                "yaw": round(s["yaw"], 6),
                "target_x": task["target_x"], "target_y": task["target_y"],
                "true_dist_m": round(dist, 3), "distance_bin": LABELS[bin_of(dist)],
                "target_in_view": int(in_view),
                "true_pixel_col": round(col, 2), "true_pixel_row": round(row, 2),
                "valid_fraction": round(float(valid), 4),
                "target_valid_fraction": task["target_valid_fraction"]})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = {"states": len(rows), "episodes": len({r["episode_id"] for r in rows}),
            "per_bin": {l: sum(1 for r in rows if r["distance_bin"] == l) for l in LABELS},
            "in_view": sum(r["target_in_view"] for r in rows),
            "source": str(STAGE06 / "episodes")}
    Path(args.meta).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
