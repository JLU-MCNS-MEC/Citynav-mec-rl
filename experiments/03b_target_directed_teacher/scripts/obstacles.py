#!/usr/bin/env python3
"""Fixed-50 m obstacle grids for the direct target-directed teacher.

The UAV flies at ``z = GROUND_LEVEL[map] + 50 m``.  A 1 m cell is blocked when
the maximum surface elevation in it reaches or exceeds that altitude, or when
the elevation is nodata / outside the raster.  The grid is built from the
elevation TIF only - never from human trajectories or markers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

DATA = Path("/home/20261024_1/Workspace/DATA")
RGBD = DATA / "rgbd"
GROUND_SRC = Path("/tmp/citynav-official/gsamllavanav/mapdata.py")
ALTITUDE_M = 50.0
CELL_M = 1.0


def load_ground() -> dict[str, float]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("official_mapdata", GROUND_SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: float(v) for k, v in module.GROUND_LEVEL.items()}


def elevation(name: str, root: Path = RGBD):
    with rasterio.open(root / f"{name}.tif") as ds:
        z = ds.read(1).astype(np.float32)
        b = ds.bounds
        nodata = ds.nodata
    valid = np.isfinite(z)
    if nodata is not None:
        valid &= z != np.float32(nodata)
    if valid.any():
        vmin = float(z[valid].min())
        if int((z == np.float32(vmin)).sum()) > 0.02 * z.size:
            # official rasterize may store nodata as min_z - 1 without a tag
            valid &= z != np.float32(vmin)
    return z, valid, b, nodata


def build(name: str, ground: dict[str, float], margin_m: float = 0.5,
          cell: float = CELL_M) -> dict:
    z, valid, b, nodata = elevation(name)
    gl = ground.get(name)
    if gl is None:
        raise KeyError(f"{name}: no GROUND_LEVEL entry")
    ceiling = float(gl) + ALTITUDE_M - margin_m
    # Only *height* is an obstacle: a nodata pixel means "no points here", not
    # a wall.  Nodata cells stay flyable (the local RGB observation is masked to
    # zero there, exactly as in the pipeline).
    blocked_px = valid & (z >= ceiling)
    h, w = blocked_px.shape
    step = max(1, int(round(cell / 0.1)))
    # pad so the decimation covers the whole raster (padding is free space)
    ph = (-h) % step
    pw = (-w) % step
    if ph or pw:
        blocked_px = np.pad(blocked_px, ((0, ph), (0, pw)), constant_values=False)
    hh, ww = blocked_px.shape
    blocked = blocked_px.reshape(hh // step, step, ww // step, step).any(axis=(1, 3))
    above = float(z[valid].max() - gl) if valid.any() else 0.0
    return {
        "name": name,
        "blocked": blocked.astype(bool),
        "cell_m": cell,
        "bounds": [float(b.left), float(b.bottom), float(b.right), float(b.top)],
        "ground_level": float(gl),
        "ceiling_m": ceiling,
        "max_above_ground_m": above,
        "blocked_ratio": float(blocked.mean()),
        "nodata": None if nodata is None else float(nodata),
    }


def load_or_build(name: str, cache: Path, ground: dict[str, float] | None = None) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    npz = cache / f"{name}.npz"
    meta_path = cache / f"{name}.json"
    if npz.exists() and meta_path.exists():
        z = np.load(npz)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["blocked"] = z["blocked"]
        return meta
    ground = ground or load_ground()
    info = build(name, ground)
    np.savez_compressed(npz, blocked=info["blocked"])
    meta = {k: v for k, v in info.items() if k != "blocked"}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return info


def world_to_cell(info: dict, x: float, y: float) -> tuple[int, int]:
    left, bottom, right, top = info["bounds"]
    cell = info["cell_m"]
    col = int((x - left) / cell)
    row = int((top - y) / cell)
    return row, col


def cell_to_world(info: dict, row: int, col: int) -> tuple[float, float]:
    left, bottom, right, top = info["bounds"]
    cell = info["cell_m"]
    return left + (col + 0.5) * cell, top - (row + 0.5) * cell


def free(info: dict, x: float, y: float) -> bool:
    blocked = info["blocked"]
    row, col = world_to_cell(info, x, y)
    if not (0 <= row < blocked.shape[0] and 0 <= col < blocked.shape[1]):
        return False
    return not bool(blocked[row, col])
