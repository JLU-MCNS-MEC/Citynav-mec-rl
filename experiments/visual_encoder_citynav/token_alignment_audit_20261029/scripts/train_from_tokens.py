#!/usr/bin/env python3
"""Train transformer-M on an existing audit token cache (no re-collection).

Used by the A/C smoke arms: identical labels, poses and splits; only the
token/history construction differs.  Never writes outside the audit cache.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[2]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("t5_smoke", ROOT / "scripts/t5_smoke.py")
t5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--model", default="transformer_m:full")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--mec-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cache = args.cache_dir
    with np.load(cache / "tokens/t5_tokens.npz") as data:
        arrays = {key: data[key] for key in data.files}
    meta = json.loads((cache / "tokens/t5_tokens.json").read_text(encoding="utf-8"))

    split = arrays["split_id"]
    train_indices = np.flatnonzero(split == 0)
    val_indices = {
        "val_seen": np.flatnonzero(split == 1),
        "val_unseen": np.flatnonzero(split == 2),
    }
    history_lengths = arrays["history_mask"].sum(1).astype(int)

    device = torch.device(args.device)
    results = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        ns = types.SimpleNamespace(
            mec_lr=float(args.lr),
            mec_weight_decay=1e-4,
            mec_batch_size=int(args.batch_size),
            mec_steps=int(args.mec_steps),
            seed=int(seed),
            cache_dir=cache,
            mec_subdir=f"mec_audit_s{seed}",
            mec_balance="none",
        )
        result = t5._train_t5_model(
            args.model, arrays, train_indices, val_indices, device, ns
        )
        result["seed"] = int(seed)
        results.append(result)
        print(json.dumps({
            "seed": seed,
            "val_seen_acc": result["final"]["val_seen"]["accuracy"],
            "val_unseen_acc": result["final"]["val_unseen"]["accuracy"],
            "vu_stop_P": result["final"]["val_unseen"]["stop_precision"],
            "vu_stop_R": result["final"]["val_unseen"]["stop_recall"],
        }, ensure_ascii=False))

    payload = {
        "cache_dir": str(cache),
        "model": args.model,
        "mec_steps": int(args.mec_steps),
        "batch_size": int(args.batch_size),
        "seeds": list(args.seeds),
        "frames": {
            "train": int(len(train_indices)),
            "val_seen": int(len(val_indices["val_seen"])),
            "val_unseen": int(len(val_indices["val_unseen"])),
        },
        "history_length": {
            "mean": float(history_lengths.mean()),
            "min": int(history_lengths.min()),
            "max": int(history_lengths.max()),
            "valid_16_fraction": float((history_lengths >= 16).mean()),
        },
        "tokens_meta_trajectories": meta.get("trajectories"),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
