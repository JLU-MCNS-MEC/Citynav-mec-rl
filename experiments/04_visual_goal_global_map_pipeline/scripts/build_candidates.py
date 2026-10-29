#!/usr/bin/env python3
"""Build fixed-grid visual candidate libraries for the 28 development maps.

Candidates are filtered by a *correct* elevation-validity mask: the centre must
be valid and at least ``--min-valid`` of the 96 m patch must be valid.  Caches
go to ``cache_v2/`` so the old (unfixed) libraries are never reused.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pipeline import DATA, CandidateLibrary, FrozenResNet, MapRaster, build_candidates

SPLITS = DATA / "sbfnav-data/original_citynav"
HERE = Path(__file__).resolve().parents[1]


def dev_maps() -> list[str]:
    names = set()
    for split in ("val_seen", "val_unseen"):
        path = SPLITS / f"citynav_{split}.json"
        for record in json.loads(path.read_text(encoding="utf-8")):
            names.add(f"{record['area']}_block_{record['block']}")
    return sorted(names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=float, default=48.0)
    parser.add_argument("--patch", type=float, default=96.0)
    parser.add_argument("--min-valid", type=float, default=0.7)
    parser.add_argument("--min-coverage", type=float, default=0.4)
    parser.add_argument("--out", type=Path, default=HERE / "cache_v2/candidates48")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    encoder = FrozenResNet(device=args.device)
    maps = dev_maps()
    total = 0
    per_map = {}
    started = time.time()
    for i, name in enumerate(maps, 1):
        out = args.out / f"{name}.npz"
        if out.exists():
            lib = CandidateLibrary.load(name, out)
            print(f"[{i}/{len(maps)}] {name}: cached {len(lib.positions)}", flush=True)
            total += len(lib.positions)
            per_map[name] = len(lib.positions)
            continue
        raster = MapRaster.load(name)
        lib = build_candidates(raster, encoder, stride=args.stride,
                               half=args.patch / 2.0, min_valid=args.min_valid,
                               min_coverage=args.min_coverage)
        lib.save(out)
        total += len(lib.positions)
        per_map[name] = len(lib.positions)
        if len(lib.positions):
            vf, cov = lib.valid_fraction, lib.coverage
            print(f"[{i}/{len(maps)}] {name}: {len(lib.positions)} candidates "
                  f"(invalid_ratio={raster.invalid_ratio:.3f}, "
                  f"valid_frac min/mean={vf.min():.2f}/{vf.mean():.2f}, "
                  f"coverage min/mean={cov.min():.2f}/{cov.mean():.2f}, "
                  f"{time.time() - started:.0f}s)", flush=True)
        else:
            print(f"[{i}/{len(maps)}] {name}: NO candidates "
                  f"(invalid_ratio={raster.invalid_ratio:.3f})", flush=True)
        del raster
    summary = {"maps": len(maps), "candidates": total,
               "stride_m": args.stride, "patch_m": args.patch,
               "min_valid": args.min_valid, "embedding_dim": 2048,
               "per_map": per_map, "seconds": round(time.time() - started, 1),
               "tag": args.tag}
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    name = "candidate_libraries_v2.json" if not args.tag else f"candidate_libraries_{args.tag}.json"
    (HERE / "results" / name).write_text(json.dumps(summary, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_map"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
