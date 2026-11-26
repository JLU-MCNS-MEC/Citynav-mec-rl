#!/usr/bin/env python3
"""Stage 17-C dual-objective training reward.

    r_t = lambda_Q * ( alpha_p * progress_t + terminal_t )
          - lambda_D * D_coll_t / D_ref

    progress_t = clip( (d_prev - d_new) / forward_m, -1, +1 )
    terminal_t = +1 on a successful terminal, -failure_penalty otherwise
    D_ref      = fixed reference cumulative service latency (ms), shared by
                 every algorithm and never derived from the current episode

Properties required by the review:

  * the Stage-16 `-legacy_e2e/100` reward is NOT used anywhere;
  * the expert checkpoints' standalone closed-loop SR is NOT used as a reward;
  * the true target coordinates are used only here, inside the environment, as
    training feedback - they never enter the agent's state;
  * progress uses the distance-to-target change, i.e. environment feedback,
    which the agent cannot observe directly (the 127-dim state contains only
    belief/match/relative features and the inherited Stage-16 features).

Hard constraints stay in the environment (feasible mask + termination).  The
energy term is NOT part of the reward: energy is a constraint, not a third
objective.  `energy_consumption_cost` is exported for logging only.

Discounting: the paper objective is the *undiscounted* sum of D_coll and the
task success rate.  With gamma < 1 the discounted return weights late latency
less.  Stage 16 used gamma = 0.99; Stage 17-C keeps that default and exposes
`gamma` in the config, with the analysis in docs/16.  Do not change it without
the review deciding so.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class DualSignals:
    terminal_success_reward: float
    navigation_progress_reward: float
    collaborative_service_latency_cost: float
    energy_consumption_cost: float
    constraint_violation: float
    total_training_reward: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RewardConfig:
    lambda_quality: float = 1.0
    lambda_latency: float = 1.0
    d_ref_ms: float = 6000.0
    progress_weight: float = 0.5
    terminal_bonus: float = 1.0
    failure_penalty: float = 0.0
    progress_clip: float = 1.0


class DualObjectiveReward:
    """Per-episode reward accumulator; call reset() at env.start()."""

    def __init__(self, cfg: RewardConfig, forward_m: float = 5.0):
        self.cfg = cfg
        self.forward_m = float(forward_m)
        self.prev_dist = None

    def reset(self, env) -> None:
        self.prev_dist = float(np.linalg.norm(np.asarray(env.pos, float)
                                              - np.asarray(env.target, float)))

    def step(self, env, info: dict, done: bool) -> DualSignals:
        cfg = self.cfg
        dist = float(np.linalg.norm(np.asarray(env.pos, float)
                                    - np.asarray(env.target, float)))
        if self.prev_dist is None:
            self.prev_dist = dist
        progress = (self.prev_dist - dist) / max(self.forward_m, 1e-9)
        progress = float(np.clip(progress, -cfg.progress_clip, cfg.progress_clip))
        self.prev_dist = dist

        terminal = 0.0
        if done:
            terminal = cfg.terminal_bonus if env.success else -cfg.failure_penalty

        nav_reward = cfg.progress_weight * progress + terminal
        d_cost = float(info.get("d_coll_ms", 0.0)) / max(cfg.d_ref_ms, 1e-9)
        e_cost = float(info.get("e_total_step_j", 0.0)) / max(
            float(getattr(env, "energy_budget_j", 1.0)), 1e-9)
        violation = 1.0 if info.get("terminated_reason") in (
            "energy_budget_exhausted", "invalid_mec_action", "no_feasible_mec") else 0.0

        total = cfg.lambda_quality * nav_reward - cfg.lambda_latency * d_cost
        return DualSignals(
            terminal_success_reward=terminal,
            navigation_progress_reward=progress,
            collaborative_service_latency_cost=d_cost,
            energy_consumption_cost=e_cost,
            constraint_violation=violation,
            total_training_reward=float(total),
        )


REWARD_EQUATIONS = {
    "objective": "J = lambda_Q * J_Q - lambda_D * J_D / D_ref",
    "per_step": "r_t = lambda_Q * (alpha_p * progress_t + terminal_t)"
                " - lambda_D * D_coll_t / D_ref",
    "progress": "clip((d_prev - d_new) / forward_m, -1, +1)",
    "terminal": "+1 success, -failure_penalty failure",
    "energy": "constraint only (feasible mask + termination), not a reward term",
}
