#!/usr/bin/env python3
"""Stage 17-A state interface.

Stage 16 state (74 dims) is kept verbatim; constraint features are appended:

    [ 0 : 74]   Stage-16 state (train16.state_features)
    [74]        normalized_remaining_energy    = remaining / initial
    [75]        normalized_consumed_energy     = cumulative / task_budget
    [76]        remaining_energy_j / 1e5       (raw quantity, scaled so the
                                                default 360 kJ budget maps to 3.6)
    [77: 87]    candidate_service_latency      (10, / d_max_ms)
    [87: 97]    candidate_energy_cost          (10, / task_budget)
    [97:107]    deadline_valid_mask            (10, 0/1)
    [107:117]   coverage_valid_mask            (10, 0/1)
    [117:127]   feasible_mec_mask              (10, 0/1)

Total 127 dims.

Causality: `candidate_service_latency` and `candidate_energy_cost` are exactly
the *pre-decision* service cost of each node (encoding + uplink + inference +
downlink, and the corresponding hover/enc/tx/rx energy).  They never contain
the expert's action or its motion energy, since that would require the MEC
inference result which does not exist yet at decision time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in [HERE, REPO / "experiments/16_cumulative_e2e_association/p1",
          REPO / "experiments/16_cumulative_e2e_association/p3"]:
    sys.path.insert(0, str(p))

from train16 import state_features   # noqa: E402  (Stage-16 74-dim state)

from config17 import Stage17Config    # noqa: E402

BASE_DIM = 74
EXTRA_DIM = 3 + 10 + 10 + 10 + 10 + 10
STATE17_DIM = BASE_DIM + EXTRA_DIM     # 127


def build_state17(env, cfg: Stage17Config) -> tuple[np.ndarray, dict]:
    base, cov_mask = state_features(env)
    masks = env.masks()
    remaining_norm = env.remaining_energy_j / max(cfg.initial_energy_j, 1e-9)
    consumed_norm = env.cumulative_energy_j / max(cfg.task_energy_budget_j, 1e-9)
    extra = np.concatenate([
        np.asarray([remaining_norm, consumed_norm,
                    env.remaining_energy_j / 1e5], np.float32),
        masks["candidate_service_latency_ms"] / max(cfg.d_max_ms, 1e-9),
        masks["candidate_energy_cost_j"] / max(cfg.task_energy_budget_j, 1e-9),
        masks["deadline_valid_mask"],
        masks["coverage_valid_mask"],
        masks["feasible_mec_mask"],
    ])
    vec = np.concatenate([base, extra]).astype(np.float32)
    if vec.shape[0] != STATE17_DIM:
        raise RuntimeError(f"state dim mismatch: {vec.shape[0]} != {STATE17_DIM}")
    meta = {
        "coverage_mask": cov_mask,
        "deadline_valid_mask": masks["deadline_valid_mask"],
        "coverage_valid_mask": masks["coverage_valid_mask"],
        "feasible_mec_mask": masks["feasible_mec_mask"],
        "candidate_service_latency_ms": masks["candidate_service_latency_ms"],
        "candidate_energy_cost_j": masks["candidate_energy_cost_j"],
        "remaining_energy_j": env.remaining_energy_j,
        "cumulative_energy_j": env.cumulative_energy_j,
    }
    return vec, meta


STATE17_LAYOUT = {
    "stage16_state": (0, 74),
    "normalized_remaining_energy": (74, 75),
    "normalized_consumed_energy": (75, 76),
    "remaining_energy_j_scaled_1e5": (76, 77),
    "candidate_service_latency": (77, 87),
    "candidate_energy_cost": (87, 97),
    "deadline_valid_mask": (97, 107),
    "coverage_valid_mask": (107, 117),
    "feasible_mec_mask": (117, 127),
}
