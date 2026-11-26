#!/usr/bin/env python3
"""Stage 17-C-R1 dual-objective training reward.

    r_t = lambda_Q * 1[t is a successful terminal]
          - lambda_D * D_coll_t / (T_max * D_max)

Summing over an episode:

    G(tau) = lambda_Q * S(tau) - lambda_D * D_sum(tau) / (T_max * D_max)

`T_max` (max navigation steps, 200) and `D_max` (per-service deadline, 400 ms)
are both read from the Stage-17 environment configuration; neither is typed
here.  Their product (80,000 ms) is a fixed normalisation constant, NOT a
cumulative-latency constraint or threshold.

Removed in R1: navigation-progress reward, progress clipping, D_ref, failure
penalty, terminal bonus.  The only two training terms are the success
indicator and the normalised per-step service latency.  Energy stays a hard
constraint of the environment; it is exported for logging only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DualSignalsR1:
    terminal_success_reward: float
    collaborative_service_latency_cost: float
    energy_consumption_cost: float
    energy_budget_violation: int
    deadline_violation: int
    total_training_reward: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RewardConfigR1:
    lambda_quality: float = 1.0
    lambda_latency: float = 1.0


class DualObjectiveRewardR1:
    """r_t = lambda_Q * success - lambda_D * D_coll / D_max (D_max from env)."""

    def __init__(self, cfg: RewardConfigR1, d_max_ms: float, t_max: int):
        if d_max_ms <= 0 or t_max <= 0:
            raise ValueError(f"invalid normaliser d_max_ms={d_max_ms}, t_max={t_max}")
        self.cfg = cfg
        self.d_max_ms = float(d_max_ms)
        self.t_max = int(t_max)
        self.scale_ms = self.d_max_ms * self.t_max

    def reset(self, env) -> None:
        """Kept for interface symmetry; R1 has no per-episode state."""
        return None

    def step(self, env, info: dict, done: bool) -> DualSignalsR1:
        cfg = self.cfg
        success = 1.0 if (done and bool(env.success)) else 0.0
        d_cost = float(info.get("d_coll_ms", 0.0)) / self.scale_ms
        e_cost = float(info.get("e_total_step_j", 0.0)) / max(
            float(getattr(env, "energy_budget_j", 1.0)), 1e-9)
        total = cfg.lambda_quality * success - cfg.lambda_latency * d_cost
        return DualSignalsR1(
            terminal_success_reward=success,
            collaborative_service_latency_cost=d_cost,
            energy_consumption_cost=e_cost,
            energy_budget_violation=int(info.get("terminated_reason")
                                        == "energy_budget_exhausted"),
            deadline_violation=int(info.get("terminated_reason")
                                   == "invalid_mec_action"
                                   and info.get("constraint_failure") == "deadline"),
            total_training_reward=float(total),
        )


REWARD_EQUATIONS = {
    "per_step": "r_t = lambda_Q*1[success] - lambda_D*D_coll_t/(T_max*D_max)",
    "episode": "G = lambda_Q*S - lambda_D*D_sum/(T_max*D_max)",
    "norm_source": "env cfg d_max_ms (400) and max_steps (200) -> 80,000 ms",
    "removed": ["D_ref", "navigation progress", "progress_weight",
                "terminal bonus", "failure penalty"],
    "energy": "hard constraint, logged only",
}
