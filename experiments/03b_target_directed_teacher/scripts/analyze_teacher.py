#!/usr/bin/env python3
"""Aggregate statistics for the direct target-directed teacher (Stage 03B).

Reads ``results/teacher_{split}.npz`` + ``.csv`` and the archived naive run
(``results_naive_15deg/``), writes tidy CSVs under ``results/analysis/`` that the
report quotes directly.

Nothing here is used to generate labels; it only describes them.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
NAIVE = HERE / "results_naive_15deg"
OUT = RESULTS / "analysis"
QUALITY = HERE.parent / "02_trajectory_quality_audit/results/train_quality.csv"
SPLITS = ("train_seen", "val_seen", "val_unseen")
ACTION_NAMES = ["Stop", "Forward", "TurnLeft", "TurnRight"]
DIST_BUCKETS = [(0.0, 20.0, "<=20"), (20.0, 30.0, "20-30"), (30.0, 50.0, "30-50"),
                (50.0, 100.0, "50-100"), (100.0, 200.0, "100-200"),
                (200.0, math.inf, ">200")]


def pct(values) -> dict:
    a = np.asarray(values, np.float64)
    if a.size == 0:
        return {}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max()), "min": float(a.min())}


def read_rows(split: str, root: Path = RESULTS) -> list[dict]:
    with (root / f"teacher_{split}.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_npz(split: str, root: Path = RESULTS):
    return np.load(root / f"teacher_{split}.npz")


def action_rows(split: str) -> list[dict]:
    rows = read_rows(split)
    groups = sorted({r["length_group"] or "failed" for r in rows})
    out = []
    for group in groups:
        subset = [r for r in rows if (r["length_group"] or "failed") == group]
        counts = Counter()
        for r in subset:
            counts["Stop"] += int(r["act_stop"])
            counts["Forward"] += int(r["act_forward"])
            counts["TurnLeft"] += int(r["act_left"])
            counts["TurnRight"] += int(r["act_right"])
        total = sum(counts.values())
        for name in ACTION_NAMES:
            out.append({"split": split, "length_group": group, "action": name,
                        "count": counts[name],
                        "share": counts[name] / total if total else 0.0,
                        "per_episode": counts[name] / len(subset) if subset else 0.0,
                        "episodes": len(subset)})
    return out


def turns_by_distance(split: str) -> list[dict]:
    z = load_npz(split)
    off, acts, states, targets = z["offsets"], z["actions"], z["states"], z["targets"]
    counts = defaultdict(Counter)
    for i in range(len(off) - 1):
        a = acts[off[i]:off[i + 1]]
        s = states[off[i]:off[i + 1]].astype(np.float64)
        t = targets[i].astype(np.float64)
        for k, act in enumerate(a):
            d = float(np.linalg.norm(s[k, :2] - t))
            for lo, hi, name in DIST_BUCKETS:
                if lo < d <= hi or (lo == 0.0 and d <= hi):
                    counts[name][int(act)] += 1
                    break
    out = []
    for _, _, name in DIST_BUCKETS:
        c = counts[name]
        total = sum(c.values())
        out.append({"split": split, "distance_bucket": name, "decisions": total,
                    **{f"{ACTION_NAMES[a].lower()}_share": (c[a] / total if total else 0.0)
                       for a in range(4)},
                    **{f"{ACTION_NAMES[a].lower()}_count": c[a] for a in range(4)}})
    return out


def summary_row(split: str, rows: list[dict]) -> dict:
    steps = np.asarray([int(r["steps"]) for r in rows], np.float64)
    turns = np.asarray([int(r["act_left"]) + int(r["act_right"]) for r in rows], np.float64)
    fwd = np.asarray([int(r["act_forward"]) for r in rows], np.float64)
    flown = np.asarray([float(r["flown_m"]) for r in rows], np.float64)
    start = np.asarray([float(r["start_target_m"]) for r in rows], np.float64)
    final = np.asarray([float(r["final_dist"]) for r in rows], np.float64)
    # tasks whose start is already inside 20 m stop immediately (flown = 0);
    # they would divide by ~0, so the overhead is only quoted on real approaches
    ideal = start - final
    real = ideal > 1.0
    s = pct(steps)
    return {
        "split": split, "episodes": len(rows),
        "success": sum(1 for r in rows if r["success"] == "True"),
        "steps_mean": s["mean"], "steps_median": s["median"],
        "steps_p95": s["p95"], "steps_p99": s["p99"], "steps_max": s["max"],
        "turns_per_episode_mean": float(turns.mean()),
        "turn_share": float(turns.sum() / steps.sum()),
        "forward_share": float(fwd.sum() / steps.sum()),
        "flown_m_mean": float(flown.mean()),
        "flown_over_ideal": float((flown[real] / ideal[real]).mean()) if real.any() else None,
        "episodes_start_inside_20m": int((~real).sum()),
        "stop_distance_mean": float(final.mean()),
        "stop_distance_p95": float(np.percentile(final, 95)),
        "stop_distance_max": float(final.max()),
    }


def chatter_compare() -> list[dict]:
    # the 20/25/30 deg sweeps were produced with
    #   run_teacher.py --cruise-trigger <N> --out-dir /tmp/03bb_t<N>
    # they are optional here: the final run + the archived naive run always exist
    variants = [("naive15", NAIVE), ("deadband25 (final)", RESULTS)]
    variants += [(f"deadband{t}", Path(f"/tmp/03bb_t{t}")) for t in (20, 30)]
    variants = [(name, root) for name, root in variants if (root / "teacher_train_seen.csv").exists()]
    out = []
    for name, root in variants:
        rows = read_rows("train_seen", root)
        steps = np.asarray([int(r["steps"]) for r in rows], np.float64)
        turns = np.asarray([int(r["act_left"]) + int(r["act_right"]) for r in rows], np.float64)
        fwd = np.asarray([int(r["act_forward"]) for r in rows], np.float64)
        flown = np.asarray([float(r["flown_m"]) if r.get("flown_m") else 5 * int(r["act_forward"])
                            for r in rows], np.float64)
        start = np.asarray([float(r["start_target_m"]) for r in rows], np.float64)
        final = np.asarray([float(r["final_dist"]) for r in rows], np.float64)
        ideal = start - final
        real = ideal > 1.0
        out.append({"variant": name, "episodes": len(rows),
                    "success": sum(1 for r in rows if r["success"] == "True"),
                    "steps_mean": float(steps.mean()), "steps_p99": float(np.percentile(steps, 99)),
                    "turns_per_episode": float(turns.mean()),
                    "turn_share": float(turns.sum() / steps.sum()),
                    "episodes_gt6_turns": float((turns > 6).mean()),
                    "flown_over_ideal": float((flown[real] / ideal[real]).mean()) if real.any() else None})
    return out


def quality_join(rows: list[dict]) -> list[dict]:
    if not QUALITY.exists():
        return []
    with QUALITY.open(encoding="utf-8") as fh:
        q = {int(r["index"]): r["quality"] for r in csv.DictReader(fh)}
    buckets = defaultdict(list)
    for r in rows:
        buckets[q.get(int(r["index"]), "missing")].append(r)
    out = []
    for name in ("high", "medium", "low"):
        sub = buckets.get(name, [])
        if not sub:
            continue
        steps = np.asarray([int(r["steps"]) for r in sub], np.float64)
        turns = np.asarray([int(r["act_left"]) + int(r["act_right"]) for r in sub], np.float64)
        flown = np.asarray([float(r["flown_m"]) for r in sub], np.float64)
        start = np.asarray([float(r["start_target_m"]) for r in sub], np.float64)
        final = np.asarray([float(r["final_dist"]) for r in sub], np.float64)
        out.append({"human_quality": name, "episodes": len(sub),
                    "steps_mean": float(steps.mean()), "steps_median": float(np.median(steps)),
                    "turns_per_episode": float(turns.mean()),
                    "flown_over_ideal": float((flown[(start - final) > 1.0] / (start - final)[(start - final) > 1.0]).mean()),
                    "length_group_share": dict(Counter(r["length_group"] for r in sub))})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", path, f"({len(rows)} rows)")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows, action_report, dist_report = [], [], []
    extras = {}
    for split in SPLITS:
        rows = read_rows(split)
        summary_rows.append(summary_row(split, rows))
        action_report += action_rows(split)
        dist_report += turns_by_distance(split)
        steps = np.asarray([int(r["steps"]) for r in rows], np.float64)
        turn = np.asarray([int(r["act_left"]) + int(r["act_right"]) for r in rows], np.float64)
        extras[split] = {
            "steps": pct(steps),
            "start_target_m": pct([float(r["start_target_m"]) for r in rows]),
            "turns_per_episode": pct(turn),
            "turns_by_length_group": {g: float(np.mean([int(r["act_left"]) + int(r["act_right"])
                                                        for r in rows if r["length_group"] == g]))
                                      for g in ("short", "medium", "long")},
            "length_group_counts": dict(Counter(r["length_group"] for r in rows)),
            "episodes_starting_inside_20m": int(sum(1 for r in rows
                                                    if float(r["start_target_m"]) <= 20.0)),
        }
    write_csv(OUT / "summary_table.csv", summary_rows)
    write_csv(OUT / "action_distribution.csv", action_report)
    write_csv(OUT / "turns_by_distance.csv", dist_report)
    compare = chatter_compare()
    write_csv(OUT / "chatter_compare.csv", compare)
    quality = quality_join(read_rows("train_seen"))
    write_csv(OUT / "quality_join_train.csv", quality)
    summary = json.loads((RESULTS / "summary.json").read_text())
    extras["teacher_config"] = summary.get("teacher_config")
    extras["length_groups"] = summary.get("length_groups")
    extras["mode_counts"] = {s: summary[s]["mode_counts"] for s in SPLITS}
    extras["obstacle_scan"] = summary["obstacle_scan"]
    extras["chatter_compare"] = compare
    extras["human_quality_join"] = quality
    (OUT / "analysis.json").write_text(json.dumps(extras, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    print("wrote", OUT / "analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
