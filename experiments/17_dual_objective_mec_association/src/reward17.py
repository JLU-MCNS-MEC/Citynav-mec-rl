#!/usr/bin/env python3
"""Stage 17-A reward / constraint-cost interface.

Stage 17-A does NOT fix the final scalar reward.  It only guarantees that the
four signals a constrained learner needs stay separable:

    navigation_reward                  Stage-16 reward (r_t = -e2e_t/100), kept
                                       unchanged for unconstrained controls
    terminal_success_reward            1.0 on success, 0 otherwise
    collaborative_service_latency_cost D_coll(t, m) / D_max       (>= 0)
    energy_consumption_cost            E_total_step / task_budget (>= 0)

and the two hard-constraint indicators:

    energy_budget_violation            1 if the budget had to block an action
    deadline_violation                 1 if an infeasible node was requested

Energy and deadline are HARD constraints: they must be enforced through
`feasible_mec_mask` / episode termination, never softened into the reward.
Training-time use (Lagrangian multipliers, cost critics, CPO, ...) is decided
in Stage 17-B and is out of scope here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from config17 import Stage17Config


@dataclass
class StepSignals:
    navigation_reward: float
    terminal_success_reward: float
    collaborative_service_latency_cost: float
    energy_consumption_cost: float
    energy_budget_violation: int
    deadline_violation: int
    terminated_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def step_signals(env, cfg: Stage17Config, navigation_reward: float,
                 d_coll_ms: float, e_total_step_j: float) -> StepSignals:
    return StepSignals(
        navigation_reward=float(navigation_reward),
        terminal_success_reward=1.0 if (env.done and env.success) else 0.0,
        collaborative_service_latency_cost=float(d_coll_ms) / max(cfg.d_max_ms, 1e-9),
        energy_consumption_cost=float(e_total_step_j) / max(cfg.task_energy_budget_j, 1e-9),
        energy_budget_violation=int(env.terminated_reason == "energy_budget_exhausted"),
        deadline_violation=int(env.constraint_failure == "deadline"),
        terminated_reason=env.terminated_reason if env.done else "",
    )


REWARD_SPEC = {
    "objective_1": "maximise task-level success rate (SR)",
    "objective_2": "minimise cumulative collaborative service latency D_sum",
    "hard_constraints": ["task energy budget E_sum <= E_budget",
                         "per-service deadline D_coll <= D_max"],
    "not_in_scope": "scalarisation weights, Lagrangian multipliers, cost critics",
}
