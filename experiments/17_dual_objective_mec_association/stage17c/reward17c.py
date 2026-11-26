#!/usr/bin/env python3
"""Stage 17-C-R1 dual-objective training reward.

    r_t = lambda_Q * 1[t is a successful terminal] - lambda_D * D_coll_t / D_max

Summing over an episode:

    G(tau) = lambda_Q * S(tau) - lambda_D * D_sum(tau) / D_max

`D_max` is the *per-service deadline* already validated in Stage 17-B
(400 ms).  It is read from the environment configuration and never re-typed
here.  There is NO cumulative-latency reference (D_ref was removed): the
accumulated normalised cost still maps monotonically to the accumulated raw
latency because D_max is a fixed constant.

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

    def __init__(self, cfg: RewardConfigR1, d_max_ms: float):
        if d_max_ms <= 0:
            raise ValueError(f"d_max_ms must be positive, got {d_max_ms}")
        self.cfg = cfg
        self.d_max_ms = float(d_max_ms)

    def reset(self, env) -> None:
        """Kept for interface symmetry; R1 has no per-episode state."""
        return None

    def step(self, env, info: dict, done: bool) -> DualSignalsR1:
        cfg = self.cfg
        success = 1.0 if (done and bool(env.success)) else 0.0
        d_cost = float(info.get("d_coll_ms", 0.0)) / self.d_max_ms
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
    "per_step": "r_t = lambda_Q * 1[success terminal] - lambda_D * D_coll_t / D_max",
    "episode": "G = lambda_Q * S - lambda_D * D_sum / D_max",
    "d_max_source": "env cfg d_max_ms (Stage 17-B validated 400 ms)",
    "removed": ["D_ref", "navigation progress", "progress_weight",
                "terminal bonus", "failure penalty"],
    "energy": "hard constraint, logged only",
}
