"""Tests for select_cases.py on synthetic alignments + episodes (no GPU/data)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from select_cases import (belief_stats, candidate_pair, greedy_select,
                          load_alignment_csv, load_episode_json, select)

# ---------------------------------------------------------------- factories


def _alignment(tmp_path: Path, rows: list[dict],
               cols: tuple[str, ...]) -> Path:
    p = tmp_path / "align.csv"
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    return p


V4V5_COLS = ("episode_id,split,map,v4_success,v5_success,v4_final_dist,"
             "v5_final_dist,v4_steps,v5_steps,v4_spl,v5_spl").split(",")


def _fake_episode(eid: str, split: str, map_name: str, success: bool,
                  failure: str, steps: int, stop_reason: str,
                  final_est_m: float, est_target_m: float,
                  max_est_m: float, stop_dist=None) -> dict:
    traj = []
    for i in range(steps):
        frac = i / max(steps - 1, 1)
        est_m = final_est_m + (max_est_m - final_est_m) * (1 - frac)
        est = [60.0 + est_target_m, 80.0] if i == steps - 1 else [60.0 + 50.0, 80.0]
        traj.append({"step": i, "x": 0.0, "y": 0.0, "yaw": 0.0, "action": 1,
                     "estimate": est, "est_distance_m": est_m,
                     "visual_score": 0.0, "accept": False, "true_dist_m": 50.0})
    return {"episode_id": eid, "split": split, "map": map_name,
            "start": [0.0, 0.0, 0.0], "target": [60.0, 80.0],
            "success": success, "failure": failure, "steps": steps,
            "stopped": success, "stop_reason": stop_reason,
            "stop_dist": stop_dist if stop_dist is not None else 10.0,
            "final_dist": 10.0 if success else 60.0, "min_dist": 9.0,
            "spl": 1.0 if success else 0.0, "trajectory": traj}


def _episode_dir(tmp_path: Path, model: str) -> dict[str, list[Path]]:
    roots = {"v4": [tmp_path / "v4_res"], "v5": [tmp_path / "v5_res"],
             "v3": [tmp_path / "v3_res"]}
    prefixes = {"v4": "full_", "v5": "v5_", "v3": "v3_"}
    for m in roots:
        (roots[m][0] / f"{prefixes[m]}val_s0" / "episodes").mkdir(parents=True)
    return roots


def _write_episode(roots: dict, model: str, rec: dict) -> Path:
    prefix = {"v4": "full_", "v5": "v5_", "v3": "v3_"}[model]
    p = roots[model][0] / f"{prefix}val_s0" / "episodes" / \
        f"{rec['episode_id'].replace(':', '_')}.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


# ------------------------------------------------------------------- tests


def test_load_alignment_csv_parses_bools_and_floats(tmp_path):
    p = _alignment(tmp_path, [
        {"episode_id": "test_unseen:00000", "split": "test_unseen",
         "map": "birmingham_block_0", "v4_success": False, "v5_success": True,
         "v4_final_dist": 99.27, "v5_final_dist": 16.35,
         "v4_steps": 84, "v5_steps": 81, "v4_spl": 0.0, "v5_spl": 0.43}],
        V4V5_COLS)
    rows = load_alignment_csv(p)
    r = rows["test_unseen:00000"]
    assert r["v4_success"] is False and r["v5_success"] is True
    assert r["v4_steps"] == 84 and abs(r["v5_final_dist"] - 16.35) < 1e-9


def test_belief_stats_winners_and_losers():
    good = _fake_episode("val_seen:00000", "val_seen", "birmingham_block_0",
                         True, "success", 20, "model_stop",
                         final_est_m=5.0, est_target_m=8.0, max_est_m=20.0)
    drifted = _fake_episode("val_seen:00000", "val_seen", "birmingham_block_0",
                            False, "out_of_bounds", 30, "out_of_bounds",
                            final_est_m=40.0, est_target_m=55.0, max_est_m=70.0)
    w, l = belief_stats(good), belief_stats(drifted)
    assert w["final_est_m"] == 5.0 and w["est_to_target_m"] == 8.0
    assert w["story_ok"] is True
    assert l["max_est_m"] == 70.0 and l["drift_ok"] is True


def test_candidate_pair_verifies_same_task(tmp_path):
    roots = _episode_dir(tmp_path, "v4")
    a = _fake_episode("val_seen:00000", "val_seen", "birmingham_block_0",
                      False, "out_of_bounds", 30, "out_of_bounds",
                      final_est_m=40.0, est_target_m=55.0, max_est_m=70.0)
    b = _fake_episode("val_seen:00000", "val_seen", "birmingham_block_0",
                      True, "success", 20, "model_stop",
                      final_est_m=5.0, est_target_m=8.0, max_est_m=20.0)
    _write_episode(roots, "v4", a)
    _write_episode(roots, "v5", b)
    w, l, err = candidate_pair("A", "val_seen:00000", roots)
    assert err == "" and w["success"] and not l["success"]
    # now corrupt the task: different target in one record
    b2 = dict(b)
    b2["target"] = [999.0, 999.0]
    _write_episode(roots, "v5", b2)
    w, l, err = candidate_pair("A", "val_seen:00000", roots)
    assert err == "task_mismatch"


def test_greedy_select_excludes_empty_candidate_map_and_collision(tmp_path):
    roots = _episode_dir(tmp_path, "v4")
    def pair(eid, map_name, failure, steps, winner_success=True,
             loser_success=False):
        w = _fake_episode(eid, "test_unseen", map_name, winner_success,
                          "success", 20, "model_stop", 5.0, 8.0, 20.0)
        l = _fake_episode(eid, "test_unseen", map_name, loser_success,
                          failure, steps, failure, 40.0, 55.0, 70.0)
        _write_episode(roots, "v5", w)
        _write_episode(roots, "v4", l)
        return eid
    ids = [pair("test_unseen:00000", "birmingham_block_0", "out_of_bounds", 30),
           pair("test_unseen:00001", "cambridge_block_32", "out_of_bounds", 31),
           pair("test_unseen:00002", "birmingham_block_1", "collision", 32),
           pair("test_unseen:00003", "birmingham_block_2", "never_stop", 33)]
    picks, notes = greedy_select("A", ids, 2, roots, exclude_empty_cand=True)
    assert {p["episode_id"] for p in picks} == {"test_unseen:00000",
                                                "test_unseen:00003"}
    assert {p["loser_failure"] for p in picks} == {"out_of_bounds",
                                                   "never_stop"}
    assert any("cambridge_block_32" in n for n in notes)
    assert any("collision" in n for n in notes)


def test_select_end_to_end_with_override(tmp_path):
    roots = _episode_dir(tmp_path, "v4")
    rows = []
    for i in range(6):
        eid = f"test_unseen:0000{i}"
        map_name = f"birmingham_block_{i}"
        _write_episode(roots, "v5", _fake_episode(
            eid, "test_unseen", map_name, True, "success", 20, "model_stop",
            5.0, 8.0, 20.0))
        _write_episode(roots, "v4", _fake_episode(
            eid, "test_unseen", map_name, False, "out_of_bounds", 30,
            "out_of_bounds", 40.0, 55.0, 70.0))
        rows.append({"episode_id": eid, "split": "test_unseen",
                     "map": map_name, "v4_success": False, "v5_success": True,
                     "v4_final_dist": 60.0, "v5_final_dist": 10.0,
                     "v4_steps": 30, "v5_steps": 20, "v4_spl": 0.0,
                     "v5_spl": 1.0})
    align = _alignment(tmp_path, rows, V4V5_COLS)
    out = tmp_path / "selection.json"
    result = select(["A"], per_category=4, count_c=2, overrides=[],
                    episode_roots=roots,
                    alignment_paths={"A": align}, out=out, seed=0)
    assert result["categories"]["A"]["pool"] == 6
    assert len(result["categories"]["A"]["selected"]) == 4
    # override forces an id even if it would not be picked by ranking
    result2 = select(["A"], per_category=2, count_c=2,
                     overrides=["test_unseen:00005"], episode_roots=roots,
                     alignment_paths={"A": align}, out=out, seed=0)
    picked = [p["episode_id"] for p in result2["categories"]["A"]["selected"]]
    assert "test_unseen:00005" in picked
    # determinism
    result3 = select(["A"], per_category=4, count_c=2, overrides=[],
                     episode_roots=roots, alignment_paths={"A": align},
                     out=out, seed=0)
    assert [p["episode_id"] for p in result3["categories"]["A"]["selected"]] == \
           [p["episode_id"] for p in result["categories"]["A"]["selected"]]


def test_load_episode_json_roundtrip(tmp_path):
    rec = _fake_episode("val_seen:00000", "val_seen", "birmingham_block_0",
                        True, "success", 3, "model_stop", 5.0, 8.0, 20.0)
    p = tmp_path / "val_seen_00000.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    assert load_episode_json(p)["episode_id"] == "val_seen:00000"
