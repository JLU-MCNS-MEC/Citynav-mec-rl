#!/usr/bin/env python3
"""Stage 10A shared constants and protocol helpers.

Protocol (identical data splits as Stage 09A):

* training:      Train-Seen teacher states (``teacher_train_seen.npz``) only,
                 EXCLUDING the 2,000-state calibration subset reproduced below
                 (the exact subset used by ``calibrate_09a.py``);
* calibration:   the same 2,000 Train-Seen states + same-map wrong goals,
                 tau grid search maximising F1 of "localised within 10 m";
* evaluation:    the committed 09A manifest (58,891 states sampled from the
                 Stage-06 closed-loop trajectories on Val-Seen/Val-Unseen),
                 joined with length_group / cohort from the Stage-06 manifest.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
A = HERE.parent
RES = A / "results"
FIG = A / "figures"

sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching/scripts"))
sys.path.insert(0, str(REPO / "experiments/04_visual_goal_global_map_pipeline/scripts"))
sys.path.insert(0, str(REPO / "experiments/05_transformer_m_training/scripts"))

import geom  # noqa: E402
import matching  # noqa: E402

TEACHER_NPZ = REPO / "experiments/03b_target_directed_teacher/results/teacher_train_seen.npz"
TEACHER_CSV = REPO / "experiments/03b_target_directed_teacher/results/teacher_train_seen.csv"
STAGE06 = REPO / "experiments/06_closed_loop_success_rate/results/seed42"
MANIFEST_09A = REPO / "experiments/09_online_visual_goal_matching/results/09a_match_manifest.csv"

BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, float("inf"))]
BIN_LABELS = ["0-20", "20-40", "40-60", "60-80", ">80"]
METHODS = ("c1", "pq", "ca")
OUT = geom.OUT
SCALE = geom.CUR_SCALE
CALIB_N = 2000
CALIB_SEED = 20261030


def require_data_root() -> None:
    if not os.environ.get("CITYNAV_DATA"):
        raise SystemExit("CITYNAV_DATA is not set (new server: "
                         "export CITYNAV_DATA=/mnt/hdd8t/Citynav-MEC/DATA)")


def load_teacher() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (maps_by_index, keys, offsets, states, targets)."""
    z = np.load(TEACHER_NPZ)
    keys = [int(k) for k in z["keys"]]
    off = z["offsets"]
    states, targets = z["states"], z["targets"]
    with TEACHER_CSV.open(encoding="utf-8") as fh:
        rows = {int(r["index"]): r for r in csv.DictReader(fh)}
    return rows, keys, off, states, targets


def calibration_subset() -> list[tuple[int, int]]:
    """Exact reproduction of the ``calibrate_09a.py`` 2,000-state pick.

    Returns the list of (episode_index, state_offset_j) pairs.  These states
    are held out from PQ/CA training and used only for tau calibration.
    """
    rows, keys, off, states, targets = load_teacher()
    cand: list[tuple[int, int]] = []
    for i, key in enumerate(keys):
        s = states[off[i]:off[i + 1]]
        t = targets[i]
        d = np.linalg.norm(s[:, :2] - t, axis=1)
        for j in np.where(d <= 48)[0]:
            cand.append((i, int(j)))
    rng = np.random.default_rng(CALIB_SEED)
    take = rng.choice(len(cand), size=min(CALIB_N, len(cand)), replace=False)
    return [cand[i] for i in take]


def bin_of(d: float) -> int:
    for i, (lo, hi) in enumerate(BINS):
        if lo <= d < hi:
            return i
    return len(BINS) - 1


def pick_steps(steps: list[tuple[int, float]], per_episode: int = 12) -> list[int]:
    """09A round-robin stratified pick over distance bins; returns step indices."""
    by_bin: dict[int, list[tuple[int, float]]] = {}
    for step, d in steps:
        by_bin.setdefault(bin_of(d), []).append((step, d))
    for b in by_bin:
        by_bin[b].sort()
    quota = max(1, per_episode // max(len(by_bin), 1))
    chosen: list[int] = []
    for b in sorted(by_bin):
        entries = by_bin[b]
        if len(entries) <= quota:
            chosen.extend(e[0] for e in entries)
        else:
            idx = sorted({round(i * (len(entries) - 1) / (quota - 1)) if quota > 1 else 0
                          for i in range(quota)})
            chosen.extend(entries[i][0] for i in idx)
    return sorted(chosen)[:per_episode]


def in_view_pixel(col: float, row: float) -> bool:
    return bool(0 <= col <= OUT - 1 and 0 <= row <= OUT - 1)


def stage06_tasks() -> dict[str, dict]:
    manifest = json.loads((STAGE06 / "manifest.json").read_text(encoding="utf-8"))
    return {t["episode_id"]: t for t in manifest["tasks"]}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
