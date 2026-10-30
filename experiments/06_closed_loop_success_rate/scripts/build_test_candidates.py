#!/usr/bin/env python3
"""Build 24 m candidate libraries for the Test-Unseen maps.

The development cache ``cache_v2/candidates24`` only covers the 28 dev maps,
so the belief cannot be computed for the 6 test maps without this.  Parameters
are identical to ``04/scripts/build_candidates.py --stride 24 --patch 96``:
stride 24 m, 96 m patch, min-valid 0.7, min-coverage 0.4, same frozen ResNet.

Output goes to a separate directory so the development cache is untouched:
    cache_v2_test/candidates24/<map>.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "04_visual_goal_global_map_pipeline/scripts"))

from pipeline import CandidateLibrary, FrozenResNet, MapRaster, build_candidates  # noqa: E402

OUT = ROOT / "04_visual_goal_global_map_pipeline/cache_v2_test/candidates24"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(
        HERE.parent / "results/test_unseen/manifest.json"))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--stride", type=float, default=24.0)
    parser.add_argument("--patch", type=float, default=96.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    names = sorted({t["map"] for t in manifest["tasks"]})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    encoder = FrozenResNet("cuda")
    started = time.time()
    total = 0
    for name in names:
        path = out / f"{name}.npz"
        if path.exists() and not args.force:
            lib = CandidateLibrary.load(name, path)
            print(f"{name}: cached {len(lib.positions)}", flush=True)
            total += len(lib.positions)
            continue
        t0 = time.time()
        lib = build_candidates(MapRaster.load(name), encoder,
                               stride=args.stride, half=args.patch / 2.0,
                               min_valid=0.7, min_coverage=0.4)
        tmp = path.with_name(f".{path.name}.tmp.npz")
        lib.save(tmp)
        tmp.replace(path)
        print(f"{name}: {len(lib.positions)} candidates in {time.time() - t0:.1f}s", flush=True)
        total += len(lib.positions)
    print(f"done: {len(names)} maps, {total} candidates, {time.time() - started:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
