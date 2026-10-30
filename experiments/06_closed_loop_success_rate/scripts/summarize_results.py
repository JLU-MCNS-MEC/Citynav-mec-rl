#!/usr/bin/env python3
"""Aggregate the Stage-06 closed-loop journal into SR/OSR/NE tables.

Reads ``results/seed42/journal.jsonl`` (one line per finished episode) and
writes ``summary.json`` plus long-format CSVs.  Every group that the protocol
asks for is emitted: split, cohort, length group, the three cross-products, map,
and the target-validity subsets.

Metrics
-------
SR        correct STOP with stop distance <= 20 m (and final distance <= 20 m)
OSR       the trajectory ever entered the 20 m radius
NE        final horizontal distance to the target
stop err  distance at the predicted STOP (stopped episodes only)
SPL       success-weighted (teacher path / max(model path, teacher path))
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results/seed42"
CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]
FAILURES = ("success", "premature_stop", "reached_but_not_stopped", "never_stop",
            "collision", "out_of_bounds", "goal_image_invalid", "other_failure")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else None


def median(xs):
    return float(sorted(xs)[len(xs) // 2]) if xs else None


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    succ = sum(1 for r in rows if r.get("success"))
    osr = sum(1 for r in rows if r.get("osr"))
    stopped = [r for r in rows if r.get("stopped")]
    failure_counts = {f: sum(1 for r in rows if r.get("failure") == f) for f in FAILURES}
    finals = [r["final_dist"] for r in rows if r.get("final_dist") is not None]
    mins = [r["min_dist"] for r in rows if r.get("min_dist") is not None]
    stop_d = [r["stop_dist"] for r in stopped if r.get("stop_dist") is not None]
    steps = [r["steps"] for r in rows if r.get("steps") is not None]
    loops = [r for r in rows if r.get("steps") == r.get("step_cap")]
    spl_terms = []
    for r in rows:
        tp, mp_ = r.get("teacher_path_len_m") or 0.0, max(r.get("path_len_m") or 0.0, 1e-9)
        if r.get("success") and tp > 0:
            spl_terms.append(tp / max(mp_, tp))
    action_totals = {c: sum(r.get("action_counts", {}).get(c, 0) for r in rows) for c in CLASSES}
    step_total = max(sum(action_totals.values()), 1)
    lo, hi = wilson(succ, n)
    return {
        "n": n, "sr": succ / n, "sr_ci95": [lo, hi], "osr": osr / n,
        "stop_rate": len(stopped) / n,
        "premature_stop_rate": failure_counts["premature_stop"] / n,
        "reached_but_not_stopped_rate": failure_counts["reached_but_not_stopped"] / n,
        "never_stop_rate": failure_counts["never_stop"] / n,
        "collision_rate": failure_counts["collision"] / n,
        "out_of_bounds_rate": failure_counts["out_of_bounds"] / n,
        "collision_any_rate": sum(1 for r in rows if r.get("collision")) / n,
        "out_of_bounds_any_rate": sum(1 for r in rows if r.get("out_of_bounds")) / n,
        "goal_image_invalid_rate": failure_counts["goal_image_invalid"] / n,
        "other_failure_rate": failure_counts["other_failure"] / n,
        "failures": failure_counts,
        "ne_mean": mean(finals), "ne_median": median(finals),
        "min_dist_mean": mean(mins), "min_dist_median": median(mins),
        "stop_error_mean": mean(stop_d), "stop_error_median": median(stop_d),
        "steps_mean": mean(steps), "steps_median": median(steps),
        "step_cap_hits": len(loops),
        "spl": mean(spl_terms),
        "spl_n": len(spl_terms),
        "action_totals": action_totals,
        "action_share": {c: action_totals[c] / step_total for c in CLASSES},
        "steps_total": step_total,
    }


def subset(rows, **kw):
    out = rows
    for key, val in kw.items():
        if val is None:
            continue
        out = [r for r in out if r.get(key) in (val if isinstance(val, (list, tuple, set)) else {val})]
    return out


def validity_subsets(rows: list[dict]) -> dict[str, list[dict]]:
    invalid = [r for r in rows if r.get("goal_image_invalid")]
    valid = [r for r in rows if not r.get("goal_image_invalid")]
    low = [r for r in valid if (r.get("target_valid_fraction") or 0.0) < 0.5]
    full = [r for r in valid if (r.get("target_valid_fraction") or 0.0) >= 0.5]
    return {"goal_image_invalid": invalid, "valid_goal_image": valid,
            "target_valid_lt_0.5": low, "target_valid_ge_0.5": full}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(RESULTS))
    args = parser.parse_args()
    res_dir = Path(args.results)
    rows = [json.loads(line) for line in
            (res_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads((res_dir / "manifest.json").read_text(encoding="utf-8"))
    task_by_id = {t["episode_id"]: t for t in manifest["tasks"]}
    for r in rows:
        t = task_by_id.get(r["episode_id"], {})
        r.setdefault("target_valid_fraction", t.get("target_valid_fraction"))
        r.setdefault("goal_image_invalid", t.get("goal_image_invalid", False))

    splits = ["val_seen", "val_unseen"]
    groups = ["short", "medium", "long"]
    scopes: dict[str, dict] = {}
    scopes["all"] = metrics(rows)
    for split in splits:
        sr = subset(rows, split=split)
        scopes[split] = metrics(sr)
        for cohort in ("A", "B"):
            scopes[f"{split}|cohort_{cohort}"] = metrics(subset(sr, cohort=cohort))
        for g in groups:
            scopes[f"{split}|{g}"] = metrics(subset(sr, length_group=g))
            for cohort in ("A", "B"):
                scopes[f"{split}|{g}|cohort_{cohort}"] = metrics(
                    subset(sr, length_group=g, cohort=cohort))
        for name, sub in validity_subsets(sr).items():
            scopes[f"{split}|{name}"] = metrics(sub)
    for g in groups:
        scopes[f"all|{g}"] = metrics(subset(rows, length_group=g))
    for cohort in ("A", "B"):
        scopes[f"all|cohort_{cohort}"] = metrics(subset(rows, cohort=cohort))

    maps = sorted({r["map"] for r in rows})
    per_map = {m: metrics([r for r in rows if r["map"] == m]) for m in maps}

    summary = {
        "tasks_in_manifest": len(manifest["tasks"]),
        "episodes_scored": len(rows),
        "checkpoint_sha": manifest["checkpoint_sha"],
        "manifest_sha": (res_dir / "manifest.sha256").read_text(encoding="utf-8").strip(),
        "step_cap": rows[0].get("step_cap") if rows else None,
        "scopes": scopes,
        "per_map": per_map,
        "failures_all": scopes["all"]["failures"],
    }
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fields = ["scope", "n", "sr", "sr_ci_lo", "sr_ci_hi", "osr", "stop_rate",
              "premature_stop_rate", "reached_but_not_stopped_rate", "never_stop_rate",
              "collision_rate", "collision_any_rate", "out_of_bounds_rate",
              "out_of_bounds_any_rate", "goal_image_invalid_rate",
              "other_failure_rate", "ne_median", "stop_error_median",
              "steps_mean", "spl", "action_stop_share", "action_forward_share",
              "action_left_share", "action_right_share"]
    with (res_dir / "summary_tables.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for scope, m in list(scopes.items()) + [(f"map|{k}", v) for k, v in per_map.items()]:
            if not m.get("n"):
                continue
            writer.writerow({
                "scope": scope, "n": m["n"], "sr": m["sr"],
                "sr_ci_lo": m["sr_ci95"][0], "sr_ci_hi": m["sr_ci95"][1],
                "osr": m["osr"], "stop_rate": m["stop_rate"],
                "premature_stop_rate": m["premature_stop_rate"],
                "reached_but_not_stopped_rate": m["reached_but_not_stopped_rate"],
                "never_stop_rate": m["never_stop_rate"],
                "collision_rate": m["collision_rate"],
                "collision_any_rate": m["collision_any_rate"],
                "out_of_bounds_rate": m["out_of_bounds_rate"],
                "out_of_bounds_any_rate": m["out_of_bounds_any_rate"],
                "goal_image_invalid_rate": m["goal_image_invalid_rate"],
                "other_failure_rate": m["other_failure_rate"],
                "ne_median": m["ne_median"], "stop_error_median": m["stop_error_median"],
                "steps_mean": m["steps_mean"], "spl": m["spl"],
                "action_stop_share": m["action_share"]["Stop"],
                "action_forward_share": m["action_share"]["Forward"],
                "action_left_share": m["action_share"]["TurnLeft"],
                "action_right_share": m["action_share"]["TurnRight"]})

    def line(name: str, m: dict) -> str:
        if not m.get("n"):
            return f"{name:28s} n=0"
        lo, hi = m["sr_ci95"]
        return (f"{name:28s} n={m['n']:5d} SR={m['sr']:.3f} [{lo:.3f},{hi:.3f}] "
                f"OSR={m['osr']:.3f} STOP={m['stop_rate']:.3f} "
                f"NEmed={m['ne_median']:.1f} steps={m['steps_mean']:.1f}")

    print(line("all", scopes["all"]))
    for split in splits:
        print(line(split, scopes[split]))
        for g in groups:
            print(line(f"  {split}|{g}", scopes[f"{split}|{g}"]))
        for cohort in ("A", "B"):
            print(line(f"  {split}|cohort_{cohort}", scopes[f"{split}|cohort_{cohort}"]))
    print("failures:", json.dumps(scopes["all"]["failures"], ensure_ascii=False))
    print("actions:", json.dumps(scopes["all"]["action_share"], ensure_ascii=False))
    print(f"wrote {res_dir / 'summary.json'} and summary_tables.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
