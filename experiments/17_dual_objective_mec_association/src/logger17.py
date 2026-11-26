#!/usr/bin/env python3
"""Stage 17-A step/episode logging.

Rules encoded here (spec section X):

  * only the *executed* node's service cost is counted in D_sum and E_sum;
  * rejected actions (infeasible node) add nothing except the violation flag;
  * once the episode is terminated no further step is charged;
  * nDTW is NOT reported: review decided (2026-11-26) that it is not needed;
    OSR is reported instead, with the 20 m success radius.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from config17 import Stage17Config


@dataclass
class StepRecord:
    episode: int
    step: int
    selected_mec: int
    selected_model: str
    navigation_action: int
    d_enc_ms: float
    d_up_ms: float
    d_inf_ms: float
    d_down_ms: float
    d_coll_ms: float
    e_hover_j: float
    e_forward_j: float
    e_turn_j: float
    e_enc_j: float
    e_tx_j: float
    e_rx_j: float
    e_total_step_j: float
    battery_remaining_j: float
    deadline_valid: bool
    energy_valid: bool
    feasible_mec_count: int


@dataclass
class EpisodeRecord:
    episode: int
    task_index: int
    map: str
    success: bool
    terminated_reason: str
    constraint_failure: str
    steps: int
    sr: int
    spl: float
    ne_m: float
    osr: int
    cumulative_collaborative_service_latency_ms: float
    legacy_cumulative_e2e_ms: float
    cumulative_energy_j: float
    remaining_energy_j: float
    energy_budget_violation_count: int
    deadline_violation_count: int
    invalid_mec_action_count: int
    model_selection_frequency: list
    service_attempt_count: int
    completed_inference_count: int
    executed_navigation_action_count: int
    service_latency_components: dict = field(default_factory=dict)


def episode_record(env, episode: int, task: dict, cfg: Stage17Config) -> EpisodeRecord:
    target = np.asarray(env.target, np.float64)
    start = np.asarray([task["start_x"], task["start_y"]], np.float64)
    l_star = float(np.linalg.norm(start - target))
    path_len = float(env.n_forward * cfg.forward_m)
    ne = float(np.linalg.norm(np.asarray(env.pos, np.float64) - target))
    spl = float(env.success) * (min(1.0, l_star / path_len) if path_len > 0 else 0.0)
    comps = {}
    for name in ("d_visual_layer4_ms", "d_visual_layer2_ms", "d_matching_ms",
                 "d_uav_total_ms", "d_up_ms", "d_inf_ms", "d_down_ms", "d_coll_ms",
                 "e_hover_j", "e_enc_j", "e_tx_j", "e_rx_j",
                 "e_forward_j", "e_turn_j"):
        comps[name] = float(sum(r.get(name, 0.0) for r in env.step_records))
    return EpisodeRecord(
        episode=episode,
        task_index=int(task.get("index", -1)),
        map=str(task["map"]),
        success=bool(env.success),
        terminated_reason=env.terminated_reason or env.failure,
        constraint_failure=env.constraint_failure,
        steps=int(env.step_idx),
        sr=int(env.success),
        spl=spl,
        ne_m=ne,
        osr=int(env.success or env.min_dist <= cfg.stop_radius_m),
        cumulative_collaborative_service_latency_ms=float(env.cumulative_coll_latency_ms),
        legacy_cumulative_e2e_ms=float(np.sum(env.episode_e2e)),
        cumulative_energy_j=float(env.cumulative_energy_j),
        remaining_energy_j=float(env.remaining_energy_j),
        energy_budget_violation_count=int(env.energy_budget_violation_count),
        deadline_violation_count=int(env.deadline_violation_count),
        invalid_mec_action_count=int(env.invalid_mec_action_count),
        model_selection_frequency=env.model_selection.astype(int).tolist(),
        service_attempt_count=int(env.service_attempt_count),
        completed_inference_count=int(env.completed_inference_count),
        executed_navigation_action_count=int(env.executed_navigation_action_count),
        service_latency_components=comps,
    )


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record) -> None:
        row = asdict(record) if hasattr(record, "__dataclass_fields__") else dict(record)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
