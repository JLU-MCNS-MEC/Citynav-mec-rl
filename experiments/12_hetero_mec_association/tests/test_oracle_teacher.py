"""Oracle teacher must reproduce the Stage-03B first action on its own path.

This is an integration test (reads rasters and the obstacle cache).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(HERE))
import protocol as P  # noqa: E402

sys.path.insert(0, str(P.REPO / "experiments/03b_target_directed_teacher/scripts"))
sys.path.insert(0, str(P.REPO / "experiments/06_closed_loop_success_rate/scripts"))
from direct_teacher import plan_path, rollout  # noqa: E402
from closed_loop import obstacle_for  # noqa: E402
from oracle_teacher import TeacherOracle  # noqa: E402


def test_oracle_matches_teacher_on_teacher_path():
    tasks = P.eval_tasks(5)
    agree = total = 0
    for t in tasks:
        info = obstacle_for(t["map"])
        pos = np.asarray([t["start_x"], t["start_y"]], np.float64)
        yaw = float(t["start_yaw"])
        tgt = np.asarray([t["target_x"], t["target_y"]], np.float64)
        wp, mode, path_len, reason = plan_path(info, pos, tgt)
        if wp is None:
            continue
        out = rollout(info, pos, yaw, tgt, wp, path_len, max_steps=200)
        orc = TeacherOracle()
        orc.reset()
        for (x, y, yy), a in zip(out["states"], out["actions"]):
            pred = orc.action(info, np.asarray([x, y], np.float64),
                              float(yy), tgt)
            agree += int(pred == a)
            total += 1
    assert total > 0
    assert agree / total >= 0.99, (agree, total)
