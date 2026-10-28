"""Verify the restored PLY set against the CityNav map requirements.

For every required map it records: file presence/size, PLY header format,
vertex count, XYZ bounds, RGB presence, SHA-256, and — when the matching
GeoTIFF is available — whether the XY bounds agree with the raster extent.
Read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.observation_adapter import open_ply_cloud  # noqa: E402

SPLITS = ("train_seen", "val_seen", "val_unseen")


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def bounds(path: Path, n_sample: int = 2_000_000) -> dict:
    cloud = open_ply_cloud(path)
    count = int(cloud.header.vertex_count)
    stride = max(1, count // n_sample)
    idx = np.arange(0, count, stride)
    data = cloud.data
    xs = np.asarray(data["x"][idx])
    ys = np.asarray(data["y"][idx])
    zs = np.asarray(data["z"][idx])
    return {
        "vertices": count,
        "x": [float(xs.min()), float(xs.max())],
        "y": [float(ys.min()), float(ys.max())],
        "z": [float(zs.min()), float(zs.max())],
        "rgb_present": all(
            name in data.dtype.names for name in ("red", "green", "blue")
        ),
        "format": cloud.header.format,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ply-root", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-sha", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ply_root = Path(args.ply_root)

    required = {}
    for split in SPLITS:
        payload = json.loads(
            (data_root / "original_citynav" / f"citynav_{split}.json").read_text()
        )
        required[split] = sorted({
            f"{r['area']}_block_{r['block']}" for r in payload
        })

    all_maps = sorted(set().union(*required.values()))
    report = {"ply_root": str(ply_root), "required": required, "maps": {}}
    for name in all_maps:
        path = ply_root / f"{name}.ply"
        entry = {"present": path.exists(), "size": None, "sha256": None, "ply": None,
                 "tif_bounds": None, "bounds_match": None}
        if path.exists():
            entry["size"] = int(path.stat().st_size)
            try:
                entry["ply"] = bounds(path)
            except Exception as exc:  # noqa: BLE001
                entry["ply_error"] = f"{type(exc).__name__}: {exc}"
            if not args.skip_sha:
                entry["sha256"] = sha256(path)
            tif = data_root / "rgbd" / f"{name}.tif"
            if tif.exists():
                import rasterio
                with rasterio.open(tif) as ds:
                    b = ds.bounds
                    entry["tif_bounds"] = [b.left, b.bottom, b.right, b.top]
                if entry["ply"]:
                    x = entry["ply"]["x"]; y = entry["ply"]["y"]
                    dev = max(
                        abs(x[0] - b.left), abs(x[1] - b.right),
                        abs(y[0] - b.bottom), abs(y[1] - b.top),
                    )
                    entry["bounds_max_deviation_m"] = round(dev, 3)
                    # One 0.1 m raster cell of rounding is expected; 1 m is a
                    # deliberately loose ceiling for sparse edge points.
                    entry["bounds_match"] = dev <= 1.0
        report["maps"][name] = entry

    missing = {s: sorted(set(required[s]) - {m for m, e in report["maps"].items() if e["present"]}) for s in SPLITS}
    report["missing"] = missing
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    for split, names in missing.items():
        print(f"{split}: {len(required[split])} required, missing {len(names)}: {names}")
    mismatched = [m for m, e in report["maps"].items() if e["bounds_match"] is False]
    print(f"bounds mismatches: {mismatched if mismatched else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
