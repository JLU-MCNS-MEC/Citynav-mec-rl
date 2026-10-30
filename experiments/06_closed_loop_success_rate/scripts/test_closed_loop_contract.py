#!/usr/bin/env python3
"""Contract tests for the Stage-06 closed-loop environment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from closed_loop import (CLASSES, FEATURE_KEYS, FORWARD_M, STOP_RADIUS, TURN_RAD,  # noqa: E402
                         classify, move, segment_status)
from dataset_v2 import history_block  # noqa: E402

RESULTS = HERE.parent / "results/seed42"
CHECKPOINT = ROOT / "05_transformer_m_training/runs_v2/seed42/best.pt"
CHECKPOINT_SHA = "077f01a4f1a643c3db7ff6a4eb57980a401232bcc4810b05925fb1fceccc0ce2"


def test_action_semantics():
    pos = np.array([10.0, 20.0])
    yaw = 0.0
    nxt, nyaw = move(pos, yaw, 1)
    assert np.allclose(nxt, [10.0 + FORWARD_M, 20.0]) and nyaw == 0.0
    nxt, nyaw = move(pos, yaw, 2)
    assert np.allclose(nxt, pos) and abs(nyaw - TURN_RAD) < 1e-12
    nxt, nyaw = move(pos, yaw, 3)
    assert np.allclose(nxt, pos) and abs(nyaw + TURN_RAD) < 1e-12
    # wrap stays in (-pi, pi]
    _, w = move(pos, math.pi - 0.01, 2)
    assert -math.pi - 1e-9 <= w <= math.pi + 1e-9


def test_segment_status_and_bounds():
    blocked = np.zeros((100, 100), bool)
    info = {"blocked": blocked, "cell_m": 1.0, "bounds": [0.0, 0.0, 100.0, 100.0]}
    assert segment_status(info, 10, 10, 15, 10) == "free"
    blocked[90, 12] = True                     # row=(top-y)/cell, col=(x-left)/cell
    assert segment_status(info, 10, 10, 15, 10) == "collision"
    blocked[90, 12] = False
    assert segment_status(info, 95, 10, 101, 10) == "out_of_bounds"


def _record(**kw):
    base = {"goal_image_invalid": False, "success": False, "collision": False,
            "out_of_bounds": False, "stopped": False, "osr": False}
    base.update(kw)
    return base


def test_failure_taxonomy():
    assert classify(_record(success=True)) == "success"
    assert classify(_record(goal_image_invalid=True)) == "goal_image_invalid"
    assert classify(_record(collision=True)) == "collision"
    assert classify(_record(collision=True, osr=True)) == "reached_but_not_stopped"
    assert classify(_record(out_of_bounds=True)) == "out_of_bounds"
    assert classify(_record(stopped=True, osr=False)) == "premature_stop"
    assert classify(_record(stopped=False, osr=True)) == "reached_but_not_stopped"
    assert classify(_record()) == "never_stop"


def test_history_is_strictly_causal_in_the_live_buffer():
    states = np.asarray([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 2.0, 0.3]], np.float32)
    actions = np.asarray([1, 2, 0], np.int64)      # last entry is a placeholder
    pooled = np.arange(3 * 8, dtype=np.float16).reshape(3, 8)
    hist = history_block(pooled.astype(np.float32), states, actions, t=2)
    assert np.allclose(hist["history_vis"][0], pooled[1].astype(np.float32))
    assert np.allclose(hist["history_vis"][1], pooled[0].astype(np.float32))
    assert hist["history_mask"].tolist() == [1, 1] + [0] * 14
    assert hist["history_action"][0].argmax() == actions[1]
    assert hist["history_action"][0, actions[2]] == 0.0


def test_success_criteria():
    # stopped inside 20 m -> success
    assert classify(_record(stopped=True, success=True, osr=True)) == "success"
    # entered 20 m but never stopped -> reached_but_not_stopped
    assert classify(_record(osr=True, stopped=False)) == "reached_but_not_stopped"


def test_checkpoint_identity():
    if not CHECKPOINT.exists():
        print("checkpoint missing, skipped")
        return
    h = hashlib.sha256()
    with CHECKPOINT.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    assert h.hexdigest() == CHECKPOINT_SHA
    assert CHECKPOINT.stat().st_size == 19563117


def test_manifest_covers_every_task_without_overlap():
    path = RESULTS / "manifest.json"
    if not path.exists():
        print("manifest missing, skipped")
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    ids = [t["episode_id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate episode ids"
    assert len(tasks) == 2498 + 2826
    for split, expect in (("val_seen", 2498), ("val_unseen", 2826)):
        sel = [t for t in tasks if t["split"] == split]
        assert len(sel) == expect
        with (ROOT / f"03b_target_directed_teacher/results/teacher_{split}.csv").open(
                encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert {int(r["index"]) for r in rows} == {t["index"] for t in sel}
        # length group must match the 03B boundary, never recomputed here
        group_of = {int(r["index"]): r["length_group"] for r in rows}
        assert all(group_of[t["index"]] == t["length_group"] for t in sel)
        cohort = {t["cohort"] for t in sel}
        assert cohort == {"A", "B"}
        for t in sel:
            assert t["cohort"] in {"A", "B"}
    # the two cohorts partition every (split, group) pool
    seen = {}
    for t in tasks:
        key = (t["split"], t["length_group"])
        seen.setdefault(key, {"A": 0, "B": 0})[t["cohort"]] += 1
    assert all(v["A"] > 0 and v["B"] > 0 for v in seen.values())


def test_target_validity_is_recorded_for_every_task():
    path = RESULTS / "manifest.json"
    if not path.exists():
        print("manifest missing, skipped")
        return
    for t in json.loads(path.read_text(encoding="utf-8"))["tasks"]:
        assert 0.0 <= t["target_valid_fraction"] <= 1.0
        assert isinstance(t["target_center_valid"], bool)
        assert isinstance(t["goal_image_invalid"], bool)
        assert t["goal_image_invalid"] == (t["target_valid_fraction"] == 0.0)


def test_model_input_keys():
    assert set(FEATURE_KEYS) == {"current", "target_emb", "history_vis", "history_pose",
                                 "history_action", "history_mask", "belief"}
    assert CLASSES == ["Stop", "Forward", "TurnLeft", "TurnRight"]
    assert STOP_RADIUS == 20.0
