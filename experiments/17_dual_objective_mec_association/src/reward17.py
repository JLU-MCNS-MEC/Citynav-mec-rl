#!/usr/bin/env python3
"""Stage 17-A-R3 reward / constraint-cost interface.

Stage 17 does NOT return the Stage-16 reward.  `env.step()` returns a scalar of
0.0 and exports every component separately, so a learner cannot silently
optimise the legacy objective.  The Stage-16 value is still available as
`legacy_stage16_reward` for historical comparison only.

Signals:

    terminal_success_reward              1.0 on success, 0 otherwise
    navigation_quality_reward            0.0 until Stage 17-C defines it
    collaborative_service_latency_cost   D_coll / D_max        (>= 0)
    energy_consumption_cost              E_step / E_budget     (>= 0)
    energy_budget_violation              1 if the budget refused an action
    deadline_violation                   1 if an infeasible node was requested
    legacy_stage16_reward                -D_legacy / 100 (comparison only)

The final scalarisation (Lagrangian, constraint ordering, Pareto) is decided in
Stage 17-C; this module only guarantees that the signals stay separable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class StepSignals:
    terminal_success_reward: float
    navigation_quality_reward: float
    collaborative_service_latency_cost: float
    energy_consumption_cost: float
    energy_budget_violation: int
    deadline_violation: int
    legacy_stage16_reward: float
    terminated_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def step_signals(info: dict) -> StepSignals:
    """Build the signal bundle from the dict returned by EpisodeEnv17.step."""
    return StepSignals(
        terminal_success_reward=float(info["terminal_success_reward"]),
        navigation_quality_reward=float(info.get("navigation_quality_reward", 0.0)),
        collaborative_service_latency_cost=float(
            info["collaborative_service_latency_cost"]),
        energy_consumption_cost=float(info["energy_consumption_cost"]),
        energy_budget_violation=int(info["energy_budget_violation"]),
        deadline_violation=int(info["deadline_violation"]),
        legacy_stage16_reward=float(info["legacy_stage16_reward"]),
        terminated_reason=str(info["terminated_reason"]),
    )


REWARD_SPEC = {
    "objective_1": "maximise task-level success rate (SR)",
    "objective_2": "minimise cumulative D_coll (D_UAV + D_up + D_inf)",
    "hard_constraints": ["episode energy <= battery capacity",
                         "per-service D_coll <= d_max_ms"],
    "default_scalar": 0.0,
    "not_in_scope": "scalarisation weights, Lagrangian multipliers, cost critics",
}
