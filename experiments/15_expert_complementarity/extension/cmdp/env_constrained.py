#!/usr/bin/env python3
"""Stage 15-Extension F6: dual-constraint (SR + energy) CMDP environment.

Independent extension: it only *subclasses* Stage 16's EpisodeEnv16 and never
modifies it.  Two constraints, both evaluated per episode:

  energy   E_episode / B_task <= 1     (B_task = slack x reference energy)
  success  failure_rate <= 1 - SR_TARGET

The reference energy is the Min-Latency baseline's energy on the *same task*,
measured at the same kinematics (v = 3 m/s), so the budget is relative and
does not depend on the absolute speed assumption.

Energy is charged from the navigation action actually executed, not from the
MEC node index (the bug that made the earlier statistics 11x too small).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
REPO = EXT.parents[2]
for p in [EXT / "energy", REPO / "experiments/16_cumulative_e2e_association/p1",
          REPO / "experiments/12_hetero_mec_association/scripts"]:
    sys.path.insert(0, str(p))

from energy_model import configure, step_energy  # noqa: E402
from env16 import EpisodeEnv16                     # noqa: E402

DEFAULT_V_FORWARD = 3.0


def load_budgets(path) -> dict:
    d = json.loads(Path(path).read_text())
    return {int(k): float(v) for k, v in d["reference_j"].items()}


class ConstrainedEnv(EpisodeEnv16):
    def __init__(self, *args, budgets=None, slack=1.2, v_forward=DEFAULT_V_FORWARD,
                 **kwargs):
        kwargs.setdefault("record_steps", True)
        super().__init__(*args, **kwargs)
        configure(v_forward=v_forward)
        self.budgets = budgets or {}
        self.slack = float(slack)
        self.energy_j = 0.0
        self.ref_energy_j = float("nan")
        self.budget_j = float("nan")
        self.has_budget = False

    def start(self, task):
        super().start(task)
        key = int(task["index"]) if "index" in task else None
        ref = self.budgets.get(key) if key is not None else None
        self.has_budget = ref is not None
        self.ref_energy_j = float(ref) if ref is not None else float("nan")
        self.budget_j = self.slack * self.ref_energy_j
        self.energy_j = 0.0

    def step(self, action):
        r, done, info = super().step(action)
        last = self.steps[-1]
        e = step_energy(int(last["action"]), self.enc_ms / 1e3,
                        last["comm_ms"] / 1e3, last["inf_ms"] / 1e3)
        self.energy_j += e["e_total_j"]
        info.update({"step_energy_j": e["e_total_j"],
                     "episode_energy_j": self.energy_j,
                     "energy_ratio": self.energy_j / max(self.budget_j, 1e-9),
                     "budget_j": self.budget_j})
        return r, done, info

    # ---- constraints ---------------------------------------------------
    def energy_ratio(self):
        return float(self.energy_j / max(self.budget_j, 1e-9))

    def energy_violation(self):
        return max(0.0, self.energy_ratio() - 1.0)

    def fail_cost(self):
        return 0.0 if self.success else 1.0

    def costs(self):
        return {"energy_ratio": self.energy_ratio(),
                "energy_violation": self.energy_violation(),
                "fail": self.fail_cost()}
