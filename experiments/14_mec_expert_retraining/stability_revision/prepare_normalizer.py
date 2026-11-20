#!/usr/bin/env python3
"""Stage 14-R2: compute Train-only normalization statistics for the three
Transformers. Writes normalization_statistics.json (shared by TS/TM/Large).

Only continuous state features are standardized:
  belief (10 dims), match (5 dims), history_pose (3 dims, valid steps only).
Visual features / target embedding come from the frozen encoder and are left
untouched. landmarks/relation already carry the V5FIX geometric normalization
(dist/50, bearing/pi); their w channel stays raw. states stay raw (world
coordinates, used only for relative geometry).

Stats are fit on the 18,151 TRAIN episodes only; the 370 Dev episodes and the
2,498 Val-Seen tasks are never seen here.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

DATA = Path("/home/jlu-mcns/citynav-ssd/11B/train_seen_memmap")
OUT = Path(__file__).resolve().parent / "normalization_statistics.json"
SEED, DEV_FRAC = 42, 0.02


def split_index() -> np.ndarray:
    ep = {int(k): tuple(v) for k, v in json.loads(
        (DATA / "ep_ranges.json").read_text()).items()}
    eps = np.asarray(sorted(ep), dtype=np.int64)
    rng = np.random.default_rng(SEED * 7777 + 14)
    perm = rng.permutation(eps)
    n_dev = max(1, int(round(len(eps) * DEV_FRAC)))
    dev = set(int(x) for x in perm[:n_dev])
    train_eps = [k for k in sorted(ep) if k not in dev]
    return np.concatenate([np.arange(ep[k][0], ep[k][1], dtype=np.int64)
                           for k in train_eps])


def moments(a: np.ndarray) -> dict:
    a64 = a.astype(np.float64)
    mean = a64.mean(axis=0)
    std = a64.std(axis=0)
    std_safe = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.tolist(), "std": std_safe.tolist(),
            "raw_std": std.tolist(),
            "min": a64.min(axis=0).tolist(), "max": a64.max(axis=0).tolist()}


def main() -> None:
    idx = split_index()
    print(f"train states: {len(idx):,}")
    belief = np.load(DATA / "belief.npy", mmap_mode="r")[idx]
    match = np.load(DATA / "match.npy", mmap_mode="r")[idx]
    hp = np.load(DATA / "hist_pose.npy", mmap_mode="r")[idx]
    hm = np.load(DATA / "hist_mask.npy", mmap_mode="r")[idx]
    valid = hm > 0.5
    print(f"history valid steps: {int(valid.sum()):,} / {valid.size:,} "
          f"({valid.mean()*100:.1f}%)")

    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(len(idx), size=min(100_000, len(idx)),
                                replace=False))
    cur = np.load(DATA / "current.npy", mmap_mode="r")[idx[sample]]
    hvis = np.load(DATA / "hist_vis.npy", mmap_mode="r")[idx[sample]]
    tgt = np.load(DATA / "target_emb.npy", mmap_mode="r")[idx[sample]]
    lm = np.load(DATA / "landmarks.npy", mmap_mode="r")[idx[sample]]
    st = np.load(DATA / "states.npy", mmap_mode="r")[idx[sample]]

    stats = {
        "meta": {
            "stage": "14-R2",
            "fit_split": "train (dev-holdout excluded)",
            "seed": SEED, "dev_frac": DEV_FRAC,
            "n_train_episodes": 18151, "n_train_states": int(len(idx)),
            "n_history_valid_steps": int(valid.sum()),
            "dev_or_val_seen_used": False,
            "applies_to": ["Transformer-S", "Transformer-M", "Large"],
        },
        "standardized": {
            "belief": {"dims": 10, "model_indices": [0, 1, 2, 3, 5, 6, 7],
                       **moments(belief)},
            "match": {"dims": 5, "model_indices": [0, 1, 2, 3, 4],
                      **moments(match)},
            "history_pose": {"dims": 3, "note": "valid steps only (hist_mask>0.5)",
                             **moments(hp[valid])},
        },
        "untouched": {
            "current": {"note": "frozen ResNet spatial features",
                        "sampled_states": int(len(sample)),
                        "mean": float(cur.mean()), "std": float(cur.std()),
                        "min": float(cur.min()), "max": float(cur.max())},
            "history_vis": {"note": "frozen ResNet pooled features",
                            "mean": float(hvis.mean()), "std": float(hvis.std()),
                            "min": float(hvis.min()), "max": float(hvis.max())},
            "target_emb": {"note": "frozen text/visual goal descriptor",
                           "mean": float(tgt.mean()), "std": float(tgt.std()),
                           "min": float(tgt.min()), "max": float(tgt.max())},
            "landmarks": {"note": "dist/50 and bearing/pi already applied in "
                                  "TransformerM12Fix.landmark_geometry; w kept raw",
                          "w_min": float(lm[..., 2].min()),
                          "w_median": float(np.median(lm[..., 2])),
                          "w_max": float(lm[..., 2].max())},
            "states": {"note": "world coordinates, only used for relative geometry",
                       "x_range": [float(st[:, 0].min()), float(st[:, 0].max())],
                       "y_range": [float(st[:, 1].min()), float(st[:, 1].max())],
                       "yaw_range": [float(st[:, 2].min()), float(st[:, 2].max())]},
            "relation": {"note": "derived from landmarks, already normalized "
                                 "to [d/50, delta/pi, valid]"},
        },
    }
    OUT.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"belief_mean": np.round(stats["standardized"]["belief"]["mean"], 3).tolist(),
                      "belief_std": np.round(stats["standardized"]["belief"]["std"], 3).tolist(),
                      "match_mean": np.round(stats["standardized"]["match"]["mean"], 3).tolist(),
                      "match_std": np.round(stats["standardized"]["match"]["std"], 3).tolist(),
                      "pose_mean": np.round(stats["standardized"]["history_pose"]["mean"], 3).tolist(),
                      "pose_std": np.round(stats["standardized"]["history_pose"]["std"], 3).tolist()},
                     indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
