#!/usr/bin/env python3
"""Split the corrected-protocol val manifest into 3 deterministic shards.

Mirrors the existing V4/V5 3-shard result layout (1775 / 1775 / 1774):
shard i holds manifest tasks i, i+3, i+6, ... in manifest order, so each
shard is resume-safe and reproducible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_shards(manifest_path: Path, out_dir: Path, num_shards: int = 3,
                prefix: str = "val_s") -> list[Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(num_shards):
        ids = [t["episode_id"] for t in tasks[i::num_shards]]
        p = out_dir / f"{prefix}{i}.txt"
        p.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        paths.append(p)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--prefix", default="val_s")
    args = parser.parse_args()
    paths = make_shards(Path(args.manifest), Path(args.out_dir),
                        args.num_shards, args.prefix)
    for p in paths:
        n = len(p.read_text(encoding="utf-8").splitlines())
        print(f"{p}  ({n} episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
