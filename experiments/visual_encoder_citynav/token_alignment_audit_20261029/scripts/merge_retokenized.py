#!/usr/bin/env python3
"""Merge re-rendered token parts into a full npz in the original row order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-cache", type=Path, required=True,
                    help="cache dir whose tokens/t5_tokens.{npz,json} define rows")
    ap.add_argument("--parts-dir", type=Path, required=True)
    ap.add_argument("--out-cache", type=Path, required=True)
    args = ap.parse_args()

    base_npz = args.base_cache / "tokens/t5_tokens.npz"
    base_meta = json.loads(
        (args.base_cache / "tokens/t5_tokens.json").read_text(encoding="utf-8")
    )
    arrays = dict(np.load(base_npz))
    frames = base_meta["frames_metadata"]
    n = len(frames)
    assert n == len(arrays["actions"]), (n, len(arrays["actions"]))

    tokens = np.zeros_like(arrays["current_tokens"])
    history = np.zeros_like(arrays["history_vis"])
    gab = np.zeros_like(arrays["gab"])
    seen = np.zeros(n, dtype=bool)

    index = {
        (f["split"], f["map_name"], f["episode_id"], int(f["teacher_step"])): i
        for i, f in enumerate(frames)
    }

    for part in sorted(args.parts_dir.glob("part_*.npz")):
        with np.load(part, allow_pickle=False) as data:
            keys = [str(k) for k in data["keys"]]
            t = data["tokens"]
            h = data["history_vis"]
            g = data["gab"]
            for j, key in enumerate(keys):
                split, map_name, episode_id, step = key.split("|")
                row = index.get((split, map_name, episode_id, int(step)))
                if row is None:
                    raise SystemExit(f"unknown re-rendered key: {key}")
                tokens[row] = t[j]
                history[row] = h[j]
                gab[row] = g[j]
                seen[row] = True

    missing = int((~seen).sum())
    if missing:
        raise SystemExit(f"{missing} of {n} rows were not re-rendered")

    arrays["current_tokens"] = tokens
    arrays["history_vis"] = history
    arrays["gab"] = gab

    out = args.out_cache / "tokens"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "t5_tokens.npz", **arrays)
    (out / "t5_tokens.json").write_text(
        json.dumps(base_meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"merged {n} rows -> {out/'t5_tokens.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
