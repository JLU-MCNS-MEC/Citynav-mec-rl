#!/usr/bin/env python3
"""Stage 15 Phase C utilities: strict navigation-state snapshots.

A snapshot stores every field required to resume the closed-loop controller
exactly where it was: UAV pose, step index, full ContinuousBelief state,
16-step history, task identity and candidate status. Everything else (map
raster, candidate library, goal embedding, matcher) is deterministically
rebuilt from the map/target.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXP14 = REPO / "experiments/14_mec_expert_retraining"
for p in [EXP14,
          REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4/scripts",
          REPO / "experiments/10_goal_query_mec_training/10D_transformer_m_training/scripts",
          REPO / "experiments/05_transformer_m_training/scripts",
          REPO / "experiments/04_visual_goal_global_map_pipeline/scripts",
          REPO / "experiments/06_closed_loop_success_rate/scripts",
          REPO / "experiments/09_online_visual_goal_matching/scripts",
          REPO / "experiments/09_online_visual_goal_matching/09C_continuous_localization/scripts"]:
    sys.path.insert(0, str(p))

from continuous_belief import ContinuousBelief  # noqa: E402


@dataclass
class Snap:
    episode_key: int
    episode_id: str
    split: str
    map: str
    target: list[float]
    step: int
    pos: list[float]
    yaw: float
    empty_cands: bool
    belief_means: np.ndarray | None
    belief_vars: np.ndarray | None
    belief_weights: np.ndarray | None
    belief_last_update: int
    belief_last_accept: np.ndarray | None
    hist_vis: np.ndarray      # (16,2048) float32
    hist_pose: np.ndarray     # (16,3) float32
    hist_act: np.ndarray      # (16,) int64
    controller: str = "ts"


def save_snapshot(path: Path, s: Snap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        belief_means=s.belief_means if s.belief_means is not None else np.zeros((0, 2)),
        belief_vars=s.belief_vars if s.belief_vars is not None else np.zeros(0),
        belief_weights=s.belief_weights if s.belief_weights is not None else np.zeros(0),
        belief_last_accept=(s.belief_last_accept if s.belief_last_accept is not None
                            else np.zeros(2)),
        hist_vis=s.hist_vis, hist_pose=s.hist_pose, hist_act=s.hist_act,
    )
    meta = {
        "episode_key": s.episode_key, "episode_id": s.episode_id,
        "split": s.split, "map": s.map, "target": s.target,
        "step": s.step, "pos": s.pos, "yaw": s.yaw,
        "empty_cands": bool(s.empty_cands),
        "belief_last_update": int(s.belief_last_update),
        "has_last_accept": bool(s.belief_last_accept is not None),
        "has_belief": bool(s.belief_means is not None and len(s.belief_means)),
        "controller": s.controller,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_snapshot(path: Path) -> Snap:
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    z = np.load(path)
    means = z["belief_means"]
    return Snap(
        episode_key=meta["episode_key"], episode_id=meta["episode_id"],
        split=meta["split"], map=meta["map"], target=meta["target"],
        step=meta["step"], pos=meta["pos"], yaw=meta["yaw"],
        empty_cands=meta["empty_cands"],
        belief_means=means if meta["has_belief"] else None,
        belief_vars=z["belief_vars"] if meta["has_belief"] else None,
        belief_weights=z["belief_weights"] if meta["has_belief"] else None,
        belief_last_update=meta["belief_last_update"],
        belief_last_accept=z["belief_last_accept"] if meta["has_last_accept"] else None,
        hist_vis=z["hist_vis"], hist_pose=z["hist_pose"], hist_act=z["hist_act"],
        controller=meta.get("controller", "ts"),
    )


def restore_belief(s: Snap) -> ContinuousBelief | None:
    if s.belief_means is None:
        return None
    b = ContinuousBelief.__new__(ContinuousBelief)
    b.means = np.asarray(s.belief_means, float)
    b.vars = np.asarray(s.belief_vars, float)
    b.weights = np.asarray(s.belief_weights, float)
    b.last_update = int(s.belief_last_update)
    b.last_accept = (None if s.belief_last_accept is None
                     else np.asarray(s.belief_last_accept, float))
    return b
