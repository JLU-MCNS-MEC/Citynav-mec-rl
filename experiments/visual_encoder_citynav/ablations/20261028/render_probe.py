"""E2 render probe: measure valid-pixel ratio and render time.

Compares the baseline renderer (splat 0, 4M points) with denser point budgets
and adaptive point size, on real CityNav frames.  Read-only: it only writes
derived caches under ``data_cache/`` and the JSON report.

Example:
  python render_probe.py --maps birmingham_block_13,cambridge_block_17 \
      --episodes-per-map 8 --out results/e2_render_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ObservationConfig, load_data_config  # noqa: E402
from src.data.citynav_dataset import CityNavDataset  # noqa: E402
from experiments.visual_encoder_citynav.data import (  # noqa: E402
    PointCloudStore,
    render_current_rgb,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ply-root", required=True)
    parser.add_argument("--maps", default="")
    parser.add_argument("--episodes-per-map", type=int, default=8)
    parser.add_argument("--points", default="4000000,8000000")
    parser.add_argument("--splats", default="0,1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="train_seen")
    args = parser.parse_args()

    if args.split == "test_unseen":
        raise PermissionError("test_unseen stays sealed")
    data_config = load_data_config(
        data_root=args.data_root, ply_root=args.ply_root
    )
    dataset = CityNavDataset(
        data_config, args.split, include_processed_descriptions=False
    )

    wanted = {m.strip() for m in args.maps.split(",") if m.strip()}
    by_map: dict[str, list] = {}
    for record in dataset.records:
        if wanted and record.map_name not in wanted:
            continue
        if not data_config.ply_path(record.map_name).exists():
            continue
        by_map.setdefault(record.map_name, []).append(record)

    results = []
    for points in [int(v) for v in args.points.split(",")]:
        store = PointCloudStore(data_config, max_points=points)
        for splat in [int(v) for v in args.splats.split(",")]:
            obs = ObservationConfig(
                image_size=336, goal_crop_size=96,
                fpv_width=336, fpv_height=336, fpv_splat_radius=splat,
            )
            ratios, times, per_map = [], [], {}
            for map_name, records in sorted(by_map.items()):
                idx = np.linspace(
                    0, len(records) - 1, min(args.episodes_per_map, len(records))
                ).astype(int)
                cloud = store.get(map_name)  # cache load outside the timer
                for i in idx:
                    record = records[int(i)]
                    pose = record.poses[len(record.poses) // 2]
                    t0 = time.perf_counter()
                    _, ratio = render_current_rgb(cloud, pose, obs)
                    dt = time.perf_counter() - t0
                    ratios.append(ratio)
                    times.append(dt)
                    per_map.setdefault(map_name, []).append(ratio)
            results.append({
                "max_points": points,
                "splat_radius": splat,
                "samples": len(ratios),
                "valid_ratio_mean": float(np.mean(ratios)) if ratios else None,
                "valid_ratio_median": float(np.median(ratios)) if ratios else None,
                "render_seconds_mean": float(np.mean(times)) if times else None,
                "per_map_mean": {
                    k: float(np.mean(v)) for k, v in sorted(per_map.items())
                },
            })
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"maps": sorted(by_map), "results": results},
                   indent=2, ensure_ascii=False) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
