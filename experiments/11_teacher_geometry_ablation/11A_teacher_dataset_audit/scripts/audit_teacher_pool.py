#!/usr/bin/env python3
"""Stage 11 / 11A: pure-human-teacher dataset audit.

Builds the M-v4 / M-v5 training pool as the intersection of

    Stage 02 High+Medium quality  ∩  Stage 03 four-action teacher success

over the ORIGINAL CityNav Train-Seen split (22,002 episodes), and audits it:

  1. final episode count
  2. total decision states
  3. per-map episodes / states
  4. short / medium / long distribution (Stage 03 global terciles 44/77 steps)
  5. trajectory length distribution
  6. four-action class ratios
  7. start-target distance distribution
  8. path efficiency (Stage 02 definition)
  9. backtrack ratio
 10. four-action teacher replay success rate

Outputs (results/):
  teacher_episode_manifest.csv   per-episode rows, original IDs + quality kept
  teacher_state_manifest.csv     per-decision-state rows (full pool, uncapped)
  teacher_pool_audit.json        all statistics + input provenance hashes
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parents[2]
RESULTS = EXP / "results"
DATA = Path("/mnt/hdd2t/Citynav-MEC/DATA/sbfnav-data/original_citynav")
S02 = REPO / "experiments/02_trajectory_quality_audit/results"
S03 = REPO / "experiments/03_fixed50m_four_action_teacher/results"

ACTION_NAMES = ("Stop", "Forward", "TurnLeft", "TurnRight")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows() -> tuple[list[dict], list[dict], list[dict], dict]:
    with (DATA / "citynav_train_seen.json").open(encoding="utf-8") as fh:
        records = json.load(fh)
    with (S02 / "train_quality.csv").open(encoding="utf-8") as fh:
        quality = list(csv.DictReader(fh))
    with (S03 / "teacher_episodes.csv").open(encoding="utf-8") as fh:
        teacher = list(csv.DictReader(fh))
    selection = json.loads((S02 / "selection_high_medium_mixed.json").read_text(encoding="utf-8"))
    assert len(records) == len(quality) == len(teacher) == 22002, (
        f"row mismatch: {len(records)}/{len(quality)}/{len(teacher)}")
    return records, quality, teacher, selection


def percentiles(v: np.ndarray, ps=(5, 25, 50, 75, 90, 95, 99)) -> dict:
    return {f"p{p}": float(np.percentile(v, p)) for p in ps}


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    records, quality, teacher, selection = load_rows()
    sel = set(selection["indices"])
    assert len(sel) == 18521
    assert set(r["index"] for r in quality) == {str(i) for i in range(22002)}

    # intersection: High+Medium ∩ teacher success (CSV stores Python bools)
    final_idx = sorted(i for i in sel
                       if str(teacher[i]["success"]).strip().lower() == "true")

    labels = np.load(S03 / "four_action_labels.npz")
    so = labels["state_offsets"]
    actions = labels["actions"]
    terc = json.loads((S03 / "summary.json").read_text())["teacher_step_terciles"]

    q = [quality[i] for i in final_idx]
    t = [teacher[i] for i in final_idx]
    steps = np.asarray([int(r["steps"]) for r in t], np.int64)
    maps = np.asarray([r["map"] for r in t])
    start_d = np.asarray([float(r["start_target_m"]) for r in t], np.float32)
    path_len = np.asarray([float(r["path_len_m"]) for r in t], np.float32)
    eff = np.asarray([float(r["efficiency"]) for r in q], np.float32)
    btr = np.asarray([float(r["backtrack_ratio"]) for r in q], np.float32)
    quality_lab = [r["quality"] for r in q]
    total_states = int(steps.sum())

    # per-episode action aggregates from the labels npz
    ep_actions = [actions[so[i]:so[i + 1]] for i in final_idx]
    action_total = Counter(int(a) for arr in ep_actions for a in arr)

    groups = np.where(steps <= terc[0], "short",
                      np.where(steps <= terc[1], "medium", "long"))
    group_counts = Counter(groups.tolist())
    group_states = {g: int(steps[groups == g].sum()) for g in ("short", "medium", "long")}

    by_map = {}
    for name in sorted(set(maps.tolist())):
        m = maps == name
        by_map[name] = {"episodes": int(m.sum()), "states": int(steps[m].sum())}

    # state manifest (full uncapped pool, ordered by episode index then t)
    state_rows = []
    for k, i in enumerate(final_idx):
        arr = ep_actions[k]
        for t_i, a in enumerate(arr):
            state_rows.append({
                "state_id": len(state_rows),
                "ep_index": i,
                "key": f"{i}|{maps[k]}",
                "map": maps[k],
                "t": t_i,
                "action": ACTION_NAMES[int(a)],
                "length_group": groups[k],
            })

    ep_rows = []
    for k, i in enumerate(final_idx):
        r = records[i]
        ep_rows.append({
            "index": i,
            "key": f"{i}|{maps[k]}",
            "map": maps[k],
            "quality": quality_lab[k],
            "teacher_success": "True",
            "steps": int(steps[k]),
            "length_group": groups[k],
            "start_target_m": round(float(start_d[k]), 4),
            "path_len_m": round(float(path_len[k]), 4),
            "efficiency": round(float(eff[k]), 6),
            "detour_ratio": quality[i]["detour_ratio"],
            "backtrack_m": quality[i]["backtrack_m"],
            "backtrack_ratio": quality[i]["backtrack_ratio"],
            "turn_total_deg": quality[i]["turn_total_deg"],
            "teacher_actions": quality[i]["teacher_actions"],
            "ground_level_m": quality[i]["ground_level_m"],
            "fixed_z_m": quality[i]["fixed_z_m"],
        })

    with (RESULTS / "teacher_episode_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ep_rows[0].keys()))
        w.writeheader()
        w.writerows(ep_rows)
    with (RESULTS / "teacher_state_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(state_rows[0].keys()))
        w.writeheader()
        w.writerows(state_rows)

    audit = {
        "generated_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {
            "citynav_train_seen.json": sha256(DATA / "citynav_train_seen.json")[:16],
            "selection_high_medium_mixed.json": sha256(S02 / "selection_high_medium_mixed.json")[:16],
            "train_quality.csv": sha256(S02 / "train_quality.csv")[:16],
            "teacher_episodes.csv": sha256(S03 / "teacher_episodes.csv")[:16],
            "four_action_labels.npz": sha256(S03 / "four_action_labels.npz")[:16],
        },
        "intersection": {
            "stage02_high_medium": 18521,
            "stage03_success": 19105,
            "final_episodes": len(final_idx),
            "excluded_by_low_quality_or_fail": 22002 - len(final_idx),
        },
        "decision_states": {"total": total_states,
                            "per_episode_mean": round(total_states / len(final_idx), 2)},
        "by_map": by_map,
        "length_groups": {"terciles": terc, "episodes": dict(group_counts),
                          "states": group_states},
        "trajectory_length": {"percentiles": percentiles(steps.astype(float))},
        "action_ratios": {ACTION_NAMES[k]: v for k, v in action_total.items()},
        "start_target_m": {"percentiles": percentiles(start_d.astype(float))},
        "path_efficiency": {"percentiles": percentiles(eff),
                            "mean": float(eff.mean())},
        "backtrack_ratio": {"percentiles": percentiles(btr),
                            "mean": float(btr.mean()),
                            "gt_0p1_episodes": int((btr > 0.1).sum())},
        "teacher_replay_success": {"overall_22002": "19105/22002 (86.83%)",
                                   "within_final_set": f"{len(final_idx)}/{len(final_idx)} (100%)"},
        "quality_mix": dict(Counter(quality_lab)),
    }
    (RESULTS / "teacher_pool_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: audit[k] for k in
                      ("intersection", "decision_states", "length_groups",
                       "action_ratios")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
