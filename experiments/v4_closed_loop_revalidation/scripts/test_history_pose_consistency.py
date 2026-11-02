#!/usr/bin/env python3
"""Unit test: training vs inference history/pose input consistency.

Takes REAL training trajectories (11B cache episode npz), walks them through
the training data path (dataset_v2.history_block) and the corrected inference
path (run_closed_loop.history_inputs) step by step, and asserts the two
produce IDENTICAL history_vis / history_pose / history_action / history_mask
tensors at every step.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments/05_transformer_m_training/scripts"))

from dataset_v2 import history_block  # noqa: E402
from run_closed_loop import HISTORY_LEN, history_inputs  # noqa: E402

CACHE = REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4" \
    / "results" / "cache" / "train_seen" / "episodes"


def test_episode(ep: int) -> bool:
    z = np.load(CACHE / f"{ep}.npz")
    pooled = z["pooled"].astype(np.float32)
    states = z["states"].astype(np.float64)
    actions = z["actions"].astype(np.int64)
    T = len(pooled)

    hist_vis: deque = deque(maxlen=HISTORY_LEN)
    hist_pose: deque = deque(maxlen=HISTORY_LEN)
    hist_act: deque = deque(maxlen=HISTORY_LEN)
    bad = 0
    for t in range(T):
        ref = history_block(pooled, states, actions, t)
        vis, pose, act, hmask = history_inputs(
            hist_vis, hist_pose, hist_act, states[t, :2], states[t, 2])
        ok = (np.allclose(vis[0], ref["history_vis"], atol=1e-5)
              and np.allclose(pose[0], ref["history_pose"], atol=1e-5)
              and np.allclose(act[0], ref["history_action"], atol=1e-5)
              and np.allclose(hmask[0], ref["history_mask"], atol=1e-5))
        if not ok:
            bad += 1
            if bad <= 2:
                print(f"  MISMATCH ep={ep} t={t}")
        # inference-side bookkeeping mirrors the live loop
        hist_vis.append(pooled[t])
        hist_pose.append((float(states[t, 0]), float(states[t, 1]),
                          float(states[t, 2])))
        hist_act.append(int(actions[t]))
    print(f"episode {ep}: {T} steps checked, {bad} mismatches")
    return bad == 0


def main() -> int:
    rng = np.random.default_rng(11)
    eps = sorted(rng.choice(len(list(CACHE.glob("*.npz"))), size=5, replace=False))
    eps = [int(p.stem) for p in CACHE.glob("*.npz")][:1] or [0]
    # pick 5 spread-out episodes deterministically
    all_eps = sorted(int(p.stem) for p in CACHE.glob("*.npz"))
    picked = [all_eps[i] for i in np.linspace(0, len(all_eps) - 1, 5, dtype=int)]
    ok_all = True
    for ep in picked:
        ok_all &= test_episode(ep)
    if not ok_all:
        print("FAIL: training/inference input mismatch")
        return 1
    print("PASS: history_vis / history_pose / history_action / history_mask "
          "identical across training and inference paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
