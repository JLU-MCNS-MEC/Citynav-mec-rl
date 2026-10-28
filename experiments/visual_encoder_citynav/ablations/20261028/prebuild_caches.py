"""Pre-build manifests and per-map caches sequentially.

Four training jobs that start at the same time would race on the same
``.npz`` cache files, so this script warms every shared artefact first:

* train / val manifests (including the multi-step E3/E4 variant);
* 4M point-cloud downsample caches;
* orthophoto candidate libraries.

After it completes, the four experiments only read these files.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ObservationConfig, load_data_config  # noqa: E402
from experiments.visual_encoder_citynav.data import (  # noqa: E402
    CandidateStore,
    PointCloudStore,
    build_manifest,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ply-root", required=True)
    ap.add_argument("--config", default=str(Path(__file__).parent / "configs" / "e1.yaml"))
    ap.add_argument("--val-samples", type=int, default=256)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_config = load_data_config(data_root=args.data_root, ply_root=args.ply_root)
    manifest_dir = PROJECT_ROOT / "data_cache" / "visual_encoder_citynav" / "manifests"

    t0 = time.perf_counter()
    steps = int(cfg.get("steps_per_trajectory", 1))
    train = build_manifest(
        data_config, "train_seen", seed=int(cfg["seed"]),
        output_dir=manifest_dir, steps_per_trajectory=steps,
    )
    print(f"train manifest: {train['records']} rows, steps={steps} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)
    if steps == 1:
        # still build the 3-step manifest used by E3/E4
        t0 = time.perf_counter()
        train3 = build_manifest(
            data_config, "train_seen", seed=42,
            output_dir=manifest_dir, steps_per_trajectory=3,
        )
        print(f"train manifest (s3): {train3['records']} rows "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    val_seen = build_manifest(
        data_config, "val_seen", max_records=args.val_samples,
        seed=int(cfg["seed"]), output_dir=manifest_dir,
    )
    val_unseen = build_manifest(
        data_config, "val_unseen", max_records=args.val_samples,
        seed=int(cfg["seed"]), output_dir=manifest_dir,
    )
    print(f"val manifests: seen={val_seen['records']} unseen={val_unseen['records']}",
          flush=True)

    maps = sorted(
        set(train["map_names"]) | set(val_seen["map_names"]) | set(val_unseen["map_names"])
    )
    print(f"warming caches for {len(maps)} maps", flush=True)

    cloud_store = PointCloudStore(
        data_config, max_points=int(cfg.get("max_points", 4_000_000))
    )
    candidate_store = CandidateStore(
        data_config,
        ObservationConfig(
            image_size=336, goal_crop_size=96,
            fpv_width=336, fpv_height=336,
            fpv_splat_radius=int(cfg.get("splat_radius", 0)),
        ),
        stride_m=float(cfg["candidate_stride_m"]),
        scales_m=tuple(float(v) for v in cfg["candidate_scales_m"]),
    )
    for i, name in enumerate(maps, 1):
        t = time.perf_counter()
        ply = data_config.ply_path(name)
        if not ply.exists():
            print(f"[{i}/{len(maps)}] {name}: MISSING PLY, skipped", flush=True)
            continue
        cloud_store.get(name)
        t_cloud = time.perf_counter() - t
        candidate_store.get(name)
        print(f"[{i}/{len(maps)}] {name}: cloud {t_cloud:.1f}s, "
              f"total {time.perf_counter()-t:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
