#!/usr/bin/env python3
"""Stage 01: audit the top-down raster and PLY assets.

Reads only ``DATA/rgbd`` (north-up RGB PNG + georeferenced height TIF),
``DATA/original_block_ply`` and the CityNav split JSONs (train/val only).
Writes ``results/map_assets.csv`` and ``results/summary.json``.

The official conventions are recorded from the cloned water-cookie/citynav
repository (``rasterize.py``, ``gsamllavanav/observation/cropclient.py``,
``gsamllavanav/dataset/mturk_trajectory.py``, ``gsamllavanav/space.py``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

DATA = Path("/home/20261024_1/Workspace/DATA")
RGBD = DATA / "rgbd"
PLY = DATA / "original_block_ply"
SPLITS = DATA / "sbfnav-data/original_citynav"
NODATA = -9999.0


def _ply_header(path: Path) -> dict:
    info = {"bytes": path.stat().st_size, "vertices": None, "faces": None}
    with path.open("rb") as fh:
        for _ in range(50):
            line = fh.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                info["vertices"] = int(line.split()[-1])
            elif line.startswith("element face"):
                info["faces"] = int(line.split()[-1])
            elif line == "end_header":
                break
    return info


def _split_maps() -> set[str]:
    """Development-split map names from Val-Seen + Val-Unseen (Test stays sealed).

    Train-Seen covers a subset of the same 24 maps, so the two validation files
    already give the complete 28-map development set without parsing the 432 MB
    Train-Seen JSON.
    """
    maps: set[str] = set()
    for split in ("val_seen", "val_unseen"):
        path = SPLITS / f"citynav_{split}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload.get("data", payload)
        for record in records:
            maps.add(record.get("map_name") or f"{record['area']}_block_{record['block']}")
    return maps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapdata", type=Path, default=Path("/tmp/citynav-official/gsamllavanav/mapdata.py"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ground: dict[str, float] = {}
    if args.mapdata.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("official_mapdata", args.mapdata)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ground = {k: float(v) for k, v in module.GROUND_LEVEL.items()}

    dev_maps = _split_maps()
    rows = []
    for tif in sorted(RGBD.glob("*.tif")):
        name = tif.stem
        png = RGBD / f"{name}.png"
        with rasterio.open(tif) as ds:
            width, height = ds.width, ds.height
            res = (round(ds.transform.a, 4), round(ds.transform.e, 4))
            origin = (round(ds.transform.c, 2), round(ds.transform.f, 2))
            z = ds.read(1)
        valid = z[z != NODATA]
        row = {
            "map": name,
            "tif": str(tif),
            "tif_bytes": tif.stat().st_size,
            "png_exists": png.exists(),
            "png_bytes": png.stat().st_size if png.exists() else 0,
            "tif_width": width,
            "tif_height": height,
            "res_x": res[0],
            "res_y": res[1],
            "origin_x": origin[0],
            "origin_y": origin[1],
            "nodata_fraction": float((z == NODATA).mean()),
            "z_min": float(valid.min()) if valid.size else None,
            "z_median": float(np.median(valid)) if valid.size else None,
            "z_max": float(valid.max()) if valid.size else None,
            "ground_level_official": ground.get(name),
            "in_dev_splits": name in dev_maps,
        }
        if png.exists():
            with Image.open(png) as im:
                row["png_width"], row["png_height"] = im.size
                row["png_mode"] = im.mode
                row["size_match"] = (im.size == (width, height))
        else:
            row["png_width"] = row["png_height"] = None
            row["png_mode"] = None
            row["size_match"] = False
        ply = PLY / f"{name}.ply"
        if ply.exists():
            row.update({f"ply_{k}": v for k, v in _ply_header(ply).items()})
        rows.append(row)

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    import csv

    with (args.out_dir / "map_assets.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    all_maps = {r["map"] for r in rows}
    summary = {
        "maps_with_tif": len(rows),
        "maps_with_png": sum(r["png_exists"] for r in rows),
        "size_mismatch": [r["map"] for r in rows if not r["size_match"]],
        "dev_split_maps": len(dev_maps),
        "dev_maps_missing_assets": sorted(dev_maps - all_maps),
        "resolution_set": sorted({(r["res_x"], r["res_y"]) for r in rows}),
        "ground_level_known": sum(r["ground_level_official"] is not None for r in rows),
        "ground_level_missing": [r["map"] for r in rows if r["ground_level_official"] is None],
        "z_min_range": [min(r["z_min"] for r in rows), max(r["z_min"] for r in rows)],
        "nodata_fraction_max": max(r["nodata_fraction"] for r in rows),
        "rgbd_bytes": sum(r["tif_bytes"] + r["png_bytes"] for r in rows),
        "ply_bytes": sum(r.get("ply_bytes", 0) or 0 for r in rows),
        "ply_missing": [r["map"] for r in rows if not r.get("ply_bytes")],
        "ground_level_table_size": len(ground),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
