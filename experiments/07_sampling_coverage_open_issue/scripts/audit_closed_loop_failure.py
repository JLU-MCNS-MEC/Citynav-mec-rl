#!/usr/bin/env python3
"""Stage-07 audit of the Stage-06 closed-loop failures (no model re-run).

Reads the stored per-step episode records and answers the protocol's questions
with stratified statistics rather than a single global action share:

1. teacher vs closed-loop action shares;
2. why turning drops;
3. what the never-turn episodes have in common;
4. target bearing / probabilities in the 5, 10, 20 steps before going OOB;
5. premature-STOP distance and probability;
6. what happens after entering 20 m without stopping;
7. why long tasks leave the map more often;
8. low-validity target patches vs success;
9. Val-Seen vs Val-Unseen differences.

Also compares the teacher state distribution with the states the model actually
visited (distance-to-target, |bearing|, turn rate), stratified by length group.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RES = ROOT / "06_closed_loop_success_rate/results/seed42"
OUT = HERE.parent / "results"
FIG = HERE.parent / "figures"
CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]
GROUPS = ("short", "medium", "long")
DIST_BUCKETS = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 10 ** 9)]
DIST_LABELS = ["0-20", "20-50", "50-100", "100-200", ">200"]
BEARING_BUCKETS = [(0, 30), (30, 60), (60, 90), (90, 180)]
BEARING_LABELS = ["0-30", "30-60", "60-90", "90-180"]


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def bearing_to(px, py, yaw, tx, ty) -> float:
    return wrap(math.atan2(ty - py, tx - px) - yaw)


def bucketize(value: float, edges: list[tuple[float, float]]) -> int:
    for i, (lo, hi) in enumerate(edges):
        if lo <= value < hi:
            return i
    return len(edges) - 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(RES))
    parser.add_argument("--sample-limit", type=int, default=0)
    args = parser.parse_args()
    res_dir = Path(args.results)
    manifest = json.loads((res_dir / "manifest.json").read_text(encoding="utf-8"))
    task_of = {t["episode_id"]: t for t in manifest["tasks"]}
    journal = [json.loads(line) for line in
               (res_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    # teacher action shares (per step) from the 03B CSVs of the same splits
    teacher_actions = Counter()
    for split in ("val_seen", "val_unseen"):
        with (ROOT / f"03b_target_directed_teacher/results/teacher_{split}.csv").open(
                encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                for key in ("stop", "forward", "left", "right"):
                    teacher_actions[key] += int(row[f"act_{key}"])
    teacher_total = sum(teacher_actions.values())
    teacher_share = {k: v / teacher_total for k, v in teacher_actions.items()}

    episodes = []
    files = sorted((res_dir / "episodes").glob("*.json"))
    if args.sample_limit:
        files = files[:args.sample_limit]
    for path in files:
        rec = json.loads(path.read_text(encoding="utf-8"))
        traj = rec.get("trajectory") or []
        t = task_of.get(rec["episode_id"], {})
        eps = {"id": rec["episode_id"], "split": rec["split"],
               "group": rec["length_group"], "map": rec["map"],
               "failure": rec["failure"], "steps": rec["steps"],
               "min_dist": rec["min_dist"], "final_dist": rec["final_dist"],
               "success": rec["success"], "osr": rec["osr"], "stopped": rec["stopped"],
               "stop_dist": rec["stop_dist"], "stop_prob": rec["stop_prob"],
               "start_dist": math.dist([t.get("start_x", 0), t.get("start_y", 0)],
                                       [t.get("target_x", 0), t.get("target_y", 0)]),
               "valid_frac": t.get("target_valid_fraction", 1.0),
               "collision": rec.get("collision"), "oob": rec.get("out_of_bounds"),
               "turns": rec["action_counts"]["TurnLeft"] + rec["action_counts"]["TurnRight"],
               "actions": rec["action_counts"], "traj": traj}
        if traj:
            eps["start_bearing"] = bearing_to(traj[0]["x"], traj[0]["y"], traj[0]["yaw"],
                                              rec["target"][0], rec["target"][1])
        else:
            eps["start_bearing"] = None
        episodes.append(eps)

    n = len(episodes)
    action_totals = {c: sum(e["actions"][c] for e in episodes) for c in CLASSES}
    step_total = sum(action_totals.values())

    def rate(sel, key):
        return sum(1 for e in sel if e[key]) / max(len(sel), 1)

    def group_stats(key):
        out = {}
        for g in GROUPS:
            sel = [e for e in episodes if e["group"] == g]
            out[g] = {"n": len(sel), "sr": rate(sel, "success"), "oob": rate(sel, "oob"),
                      "premature": sum(1 for e in sel if e["failure"] == "premature_stop") / max(len(sel), 1),
                      "turns_per_step": sum(e["turns"] for e in sel) /
                                        max(sum(e["steps"] for e in sel), 1),
                      "mean_start_dist": float(np.mean([e["start_dist"] for e in sel]))}
        return out

    # 1-2 action shares, turn deficit by group
    turn_share = (action_totals["TurnLeft"] + action_totals["TurnRight"]) / max(step_total, 1)
    teacher_turn = teacher_share["left"] + teacher_share["right"]

    # 3 never-turn episodes
    never = [e for e in episodes if e["turns"] == 0]
    turned = [e for e in episodes if e["turns"] > 0]

    def profile(sel):
        return {"n": len(sel), "sr": rate(sel, "success"), "oob": rate(sel, "oob"),
                "mean_start_dist": float(np.mean([e["start_dist"] for e in sel])),
                "mean_min_dist": float(np.mean([e["min_dist"] for e in sel])),
                "mean_abs_start_bearing": float(np.mean([abs(e["start_bearing"])
                                                         for e in sel if e["start_bearing"] is not None])),
                "share_bearing_gt90": sum(1 for e in sel if e["start_bearing"] is not None
                                          and abs(e["start_bearing"]) > math.pi / 2) / max(len(sel), 1),
                "map_top": Counter(e["map"] for e in sel).most_common(3)}

    # 4 pre-OOB behaviour
    pre_oob = {k: {"abs_bearing": [], "p_turn": [], "turn_action": [], "p_forward": []}
               for k in (5, 10, 20)}
    for e in episodes:
        if not e["oob"]:
            continue
        traj = e["traj"]
        for k in pre_oob:
            if len(traj) <= k:
                continue
            st = traj[-1 - k]
            # target bearing at that state (target from the last record frame is not stored;
            # use the manifest)
            t = task_of[e["id"]]
            b = bearing_to(st["x"], st["y"], st["yaw"], t["target_x"], t["target_y"])
            pre_oob[k]["abs_bearing"].append(abs(b))
            pre_oob[k]["p_turn"].append(st["probs"][2] + st["probs"][3])
            pre_oob[k]["p_forward"].append(st["probs"][1])
            pre_oob[k]["turn_action"].append(int(st["action"] in (2, 3)))
    pre_oob_summary = {f"{k}_steps": {"n": len(v["abs_bearing"]),
                                      "mean_abs_bearing_deg": float(np.degrees(np.mean(v["abs_bearing"]))) if v["abs_bearing"] else None,
                                      "mean_p_turn": float(np.mean(v["p_turn"])) if v["p_turn"] else None,
                                      "actual_turn_rate": float(np.mean(v["turn_action"])) if v["turn_action"] else None,
                                      "mean_p_forward": float(np.mean(v["p_forward"])) if v["p_forward"] else None}
                        for k, v in pre_oob.items()}

    # 5 premature stops
    pre = [e for e in episodes if e["failure"] == "premature_stop"]
    premature = {"n": len(pre),
                 "stop_dist_mean": float(np.mean([e["stop_dist"] for e in pre])) if pre else None,
                 "stop_dist_median": float(np.median([e["stop_dist"] for e in pre])) if pre else None,
                 "stop_prob_mean": float(np.mean([e["stop_prob"] for e in pre])) if pre else None,
                 "by_group": {g: sum(1 for e in pre if e["group"] == g) for g in GROUPS}}

    # 6 reached but not stopped
    rv = [e for e in episodes if e["failure"] == "reached_but_not_stopped"]
    reached = {"n": len(rv),
               "stopped_late": sum(1 for e in rv if e["stopped"]),
               "final_dist_median": float(np.median([e["final_dist"] for e in rv])) if rv else None,
               "min_dist_median": float(np.median([e["min_dist"] for e in rv])) if rv else None,
               "by_group": {g: sum(1 for e in rv if e["group"] == g) for g in GROUPS}}

    # 7 long tasks and OOB by start distance
    oob_by_start = {}
    for i, label in enumerate(DIST_LABELS):
        sel = [e for e in episodes if DIST_BUCKETS[i][0] <= e["start_dist"] < DIST_BUCKETS[i][1]]
        oob_by_start[label] = {"n": len(sel), "oob_rate": rate(sel, "oob"), "sr": rate(sel, "success")}
    oob_by_group = {g: {"n": sum(1 for e in episodes if e["group"] == g),
                        "oob_rate": rate([e for e in episodes if e["group"] == g], "oob")}
                    for g in GROUPS}

    # 8 target validity
    valid_sets = {"<0.25": [e for e in episodes if 0 < e["valid_frac"] < 0.25],
                  "0.25-0.5": [e for e in episodes if 0.25 <= e["valid_frac"] < 0.5],
                  "0.5-0.75": [e for e in episodes if 0.5 <= e["valid_frac"] < 0.75],
                  ">=0.75": [e for e in episodes if e["valid_frac"] >= 0.75],
                  "==0": [e for e in episodes if e["valid_frac"] == 0]}
    validity = {k: {"n": len(v), "sr": rate(v, "success"),
                    "osr": rate(v, "osr"),
                    "stop_rate": rate(v, "stopped")} for k, v in valid_sets.items()}

    # 9 seen vs unseen
    seen_unseen = {}
    for split in ("val_seen", "val_unseen"):
        sel = [e for e in episodes if e["split"] == split]
        seen_unseen[split] = {"n": len(sel), "sr": rate(sel, "success"),
                              "mean_steps": float(np.mean([e["steps"] for e in sel])),
                              "mean_start_dist": float(np.mean([e["start_dist"] for e in sel])),
                              "mean_valid_frac": float(np.mean([e["valid_frac"] for e in sel])),
                              "maps": len(set(e["map"] for e in sel)),
                              "groups": {g: sum(1 for e in sel if e["group"] == g) for g in GROUPS}}

    # distribution shift: teacher states vs model-visited states
    teacher = np.load(ROOT / "03b_target_directed_teacher/results/teacher_train_seen.npz")
    t_states, t_actions, t_targets, t_offsets = (teacher["states"], teacher["actions"],
                                                 teacher["targets"], teacher["offsets"])
    ep_idx = np.repeat(np.arange(len(t_offsets) - 1), np.diff(t_offsets))
    t_dist = np.linalg.norm(t_states[:, :2] - t_targets[ep_idx], axis=1)
    t_bearing = np.abs([bearing_to(s[0], s[1], s[2], tg[0], tg[1])
                        for s, tg in zip(t_states, t_targets[ep_idx])])
    t_dist_hist = np.bincount(np.digitize(t_dist, [b[1] for b in DIST_BUCKETS[:-1]], right=True),
                              minlength=5) / len(t_dist)
    t_bearing_hist = np.bincount(np.digitize(t_bearing, [b[1] * math.pi / 180
                                                         for b in BEARING_BUCKETS[:-1]], right=True),
                                 minlength=4) / len(t_bearing)

    m_dist, m_bearing, m_turn = [], [], []
    for e in episodes:
        for st in e["traj"]:
            m_dist.append(st["dist"])
            t = task_of[e["id"]]
            m_bearing.append(abs(bearing_to(st["x"], st["y"], st["yaw"],
                                            t["target_x"], t["target_y"])))
            m_turn.append(int(st["action"] in (2, 3)))
    m_dist_hist = np.bincount(np.digitize(m_dist, [b[1] for b in DIST_BUCKETS[:-1]], right=True),
                              minlength=5) / len(m_dist)
    m_bearing_hist = np.bincount(np.digitize(m_bearing, [b[1] * math.pi / 180
                                                         for b in BEARING_BUCKETS[:-1]], right=True),
                                 minlength=4) / len(m_bearing)

    result = {
        "episodes": n,
        "action_share": {"teacher": teacher_share,
                         "closed_loop": {c: action_totals[c] / max(step_total, 1) for c in CLASSES},
                         "teacher_turn": teacher_turn, "closed_loop_turn": turn_share},
        "turns_per_step_by_group": group_stats("group"),
        "never_turn": {"n": len(never), "share": len(never) / n, "profile": profile(never),
                       "turned_profile": profile(turned)},
        "pre_oob": pre_oob_summary,
        "premature_stop": premature,
        "reached_but_not_stopped": reached,
        "oob_by_start_distance": oob_by_start,
        "oob_by_group": oob_by_group,
        "target_validity": validity,
        "seen_vs_unseen": seen_unseen,
        "distribution_shift": {
            "teacher_dist_hist": t_dist_hist.tolist(),
            "closed_dist_hist": m_dist_hist.tolist(),
            "teacher_bearing_hist": t_bearing_hist.tolist(),
            "closed_bearing_hist": m_bearing_hist.tolist(),
            "dist_labels": DIST_LABELS,
            "bearing_labels": BEARING_LABELS,
            "teacher_turn_rate": float((np.isin(t_actions, [2, 3])).mean()),
            "closed_turn_rate": float(np.mean(m_turn)),
            "teacher_states": int(len(t_states)), "closed_states": len(m_dist),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "failure_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")

    # CSV of the main stratified tables
    with (OUT / "failure_tables.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "key", "n", "sr", "oob_rate", "turn_rate", "extra"])
        for g, v in result["turns_per_step_by_group"].items():
            w.writerow(["length_group", g, v["n"], f"{v['sr']:.6f}", f"{v['oob']:.6f}",
                        f"{v['turns_per_step']:.6f}", f"start_dist={v['mean_start_dist']:.1f}"])
        for k, v in oob_by_start.items():
            w.writerow(["start_distance", k, v["n"], f"{v['sr']:.6f}", f"{v['oob_rate']:.6f}",
                        "", ""])
        for k, v in validity.items():
            w.writerow(["target_validity", k, v["n"], f"{v['sr']:.6f}", "", f"{v['stop_rate']:.6f}",
                        f"osr={v['osr']:.6f}"])
        for split, v in seen_unseen.items():
            w.writerow(["split", split, v["n"], f"{v['sr']:.6f}", "", "",
                        f"steps={v['mean_steps']:.1f};start={v['mean_start_dist']:.1f}"])

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0][0]
    keys = ["stop", "forward", "left", "right"]
    xs = np.arange(4)
    ax.bar(xs - 0.2, [teacher_share[k] for k in keys], 0.4, label="teacher (03B)")
    ax.bar(xs + 0.2, [action_totals[c] / step_total for c in CLASSES], 0.4, label="closed loop")
    ax.set_xticks(xs); ax.set_xticklabels(CLASSES, fontsize=8); ax.set_ylim(0, 1)
    ax.set_title("Action share: teacher states vs model-executed", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[0][1]
    ax.bar(DIST_LABELS, [oob_by_start[k]["oob_rate"] for k in DIST_LABELS], color="#b23b3b")
    ax.set_ylim(0, 1); ax.set_title("Out-of-bounds rate by start distance", fontsize=9)
    ax.set_xlabel("start distance to target (m)", fontsize=8)
    ax = axes[1][0]
    ax.hist([e["stop_dist"] for e in pre], bins=np.arange(20, 70, 2), color="#3b6ea5")
    ax.axvline(20, color="k", ls="--", lw=1)
    ax.set_title(f"Premature STOP distance (n={len(pre)})", fontsize=9)
    ax.set_xlabel("m", fontsize=8)
    ax = axes[1][1]
    w = 0.35
    ax.bar(np.arange(4) - w / 2, result["distribution_shift"]["teacher_bearing_hist"], w,
           label="teacher states")
    ax.bar(np.arange(4) + w / 2, result["distribution_shift"]["closed_bearing_hist"], w,
           label="closed-loop states")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{b}°" for b in BEARING_LABELS], fontsize=8)
    ax.set_title("Distribution shift: |bearing| to target", fontsize=9)
    ax.set_xlabel("angle between heading and target", fontsize=8)
    ax.set_ylabel("share of states", fontsize=8); ax.legend(fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "failure_audit.png", dpi=150)
    plt.close(fig)

    print(json.dumps({k: result[k] for k in ("action_share", "never_turn", "pre_oob",
                                             "premature_stop", "reached_but_not_stopped",
                                             "oob_by_group", "seen_vs_unseen")},
                     ensure_ascii=False, indent=1))
    print("validity:", json.dumps(validity, ensure_ascii=False))
    print("shift:", json.dumps({k: result["distribution_shift"][k] for k in
                                ("teacher_dist_hist", "closed_dist_hist",
                                 "teacher_bearing_hist", "closed_bearing_hist",
                                 "teacher_turn_rate", "closed_turn_rate")}, ensure_ascii=False))
    print(f"wrote {OUT / 'failure_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
