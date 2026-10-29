#!/usr/bin/env python3
"""Build fixed-grid visual candidate libraries for the 28 development maps."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from pipeline import DATA, RGBD, CandidateLibrary, FrozenResNet, MapRaster, build_candidates

SPLITS = DATA / "sbfnav-data/original_citynav"
HERE = Path(__file__).resolve().parents[1]
CACHE = HERE / "cache/candidates"


def dev_maps() -> list[str]:
    names = set()
    for split in ("val_seen", "val_unseen"):
        path = SPLITS / f"citynav_{split}.json"
        for record in json.loads(path.read_text(encoding="utf-8")):
            names.add(f"{record['area']}_block_{record['block']}")
    return sorted(names)


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    encoder = FrozenResNet(device="cpu")
    maps = dev_maps()
    total = 0
    started = time.time()
    for i, name in enumerate(maps, 1):
        out = CACHE / f"{name}.npz"
        if out.exists():
            lib = CandidateLibrary.load(name, out)
            print(f"[{i}/{len(maps)}] {name}: cached {len(lib.positions)}", flush=True)
            total += len(lib.positions)
            continue
        raster = MapRaster.load(name)
        lib = build_candidates(raster, encoder)
        lib.save(out)
        total += len(lib.positions)
        print(f"[{i}/{len(maps)}] {name}: {len(lib.positions)} candidates "
              f"({time.time() - started:.0f}s)", flush=True)
        del raster
    summary = {"maps": len(maps), "candidates": total, "stride_m": 48.0,
               "patch_m": 96.0, "embedding_dim": 2048,
               "seconds": round(time.time() - started, 1)}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/candidate_libraries.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
