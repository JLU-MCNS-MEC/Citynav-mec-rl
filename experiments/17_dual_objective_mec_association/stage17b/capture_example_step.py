#!/usr/bin/env python3
"""Stage 17-B: capture one real decision step for the smoke report.

Runs a short Min-Latency episode on one Train task, counts the UAV encoder and
matching-encoder calls, and dumps the complete latency/energy breakdown of one
step plus the episode totals.  No training, no evaluation split.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
for p in [ROOT / "src", REPO / "experiments/16_cumulative_e2e_association/p1",
          REPO / "experiments/16_cumulative_e2e_association/p2",
          REPO / "experiments/12_hetero_mec_association/scripts"]:
    sys.path.insert(0, str(p))

from baselines import choose                                   # noqa: E402
from config17 import load_config                               # noqa: E402
from env16 import build_deployment_by_map, load_models, load_norm  # noqa: E402
from env17 import EpisodeEnv17                                 # noqa: E402
from train_association_rl import load_tasks_and_split          # noqa: E402


class _C:
    def __init__(self, obj):
        self.obj, self.calls = obj, 0

    def __call__(self, *a, **kw):
        self.calls += 1
        return self.obj(*a, **kw)

    def __getattr__(self, n):
        return getattr(self.obj, n)


class _FeatCounter:
    """matching.Encoder is used through .features(), not as a callable."""

    def __init__(self, enc):
        self.enc, self.calls = enc, 0

    def features(self, *a, **kw):
        self.calls += 1
        return self.enc.features(*a, **kw)

    def __getattr__(self, n):
        return getattr(self.enc, n)


def main():
    torch.set_num_threads(4)
    cfg = load_config(ROOT / "configs/default.json")
    train_tasks, _ = load_tasks_and_split()
    task = train_tasks[0]
    env = EpisodeEnv17(build_deployment_by_map(), "cuda", load_models("cuda"),
                       load_norm(), cfg)
    enc = _C(env.enc4)
    matcher = _FeatCounter(env.matcher_enc)
    env.enc4, env.matcher_enc = enc, matcher
    env.start(task)
    rng = np.random.default_rng(7)
    done = False
    while not done:
        a = choose("min_latency", env.metrics(), rng)
        _, done, _ = env.step(a)
    recs = env.step_records
    # pick the first step that actually completed a service and moved forward
    pick = next((r for r in recs if r["navigation_action"] == 1 and r["d_coll_ms"] > 0),
                recs[0])
    out = {
        "task_index": int(task["index"]), "map": task["map"],
        "encoder_calls": enc.calls, "matching_encoder_calls": matcher.calls,
        "steps": int(env.step_idx), "records": len(recs),
        "success": bool(env.success), "failure": env.failure,
        "example_step": pick,
        "episode": {
            "cumulative_collaborative_service_latency_ms":
                float(env.cumulative_coll_latency_ms),
            "legacy_cumulative_e2e_ms": float(np.sum(env.episode_e2e)),
            "cumulative_energy_j": float(env.cumulative_energy_j),
            "remaining_energy_j": float(env.remaining_energy_j),
            "energy_budget_j": float(env.energy_budget_j),
            "service_attempt_count": int(env.service_attempt_count),
            "completed_inference_count": int(env.completed_inference_count),
            "executed_navigation_action_count":
                int(env.executed_navigation_action_count),
            "sum_step_energy_j": float(sum(r["e_total_step_j"] for r in recs)),
            "sum_step_latency_ms": float(sum(r["d_coll_ms"] for r in recs)),
            "model_selection": env.model_selection.astype(int).tolist(),
        },
    }
    path = ROOT / "smoke/example_step.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
