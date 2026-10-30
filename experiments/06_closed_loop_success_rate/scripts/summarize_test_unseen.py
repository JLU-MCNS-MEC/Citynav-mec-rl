#!/usr/bin/env python3
"""Summarize the Test-Unseen closed-loop run (no length groups / SPL).

There are no 03B teacher labels for Test-Unseen, so the report cannot use the
Train-Seen length groups or SPL.  It stratifies by map, by start distance, by
cohort and by target-image validity instead, reusing the metric definitions of
``summarize_results.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from summarize_results import FAILURES, metrics, subset, validity_subsets, wilson  # noqa: E402

DIST_BUCKETS = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 10 ** 9)]
DIST_LABELS = ["0-20", "20-50", "50-100", "100-200", ">200"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(HERE.parent / "results/test_unseen"))
    args = parser.parse_args()
    res = Path(args.results)
    rows = [json.loads(line) for line in
            (res / "journal.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads((res / "manifest.json").read_text(encoding="utf-8"))
    tasks = {t["episode_id"]: t for t in manifest["tasks"]}
    step_cap = 200
    cfg = res / "eval_config.json"
    if cfg.exists():
        step_cap = int(json.loads(cfg.read_text(encoding="utf-8")).get("step_cap", 200))
    for r in rows:
        t = tasks[r["episode_id"]]
        r.setdefault("step_cap", step_cap)
        r["start_dist"] = math.dist([t["start_x"], t["start_y"]],
                                    [t["target_x"], t["target_y"]])
        r.setdefault("target_valid_fraction", t["target_valid_fraction"])
        r.setdefault("goal_image_invalid", t["goal_image_invalid"])
        r["dist_bucket"] = next(label for (lo, hi), label in zip(DIST_BUCKETS, DIST_LABELS)
                                if lo <= r["start_dist"] < hi)

    scopes = {"all": metrics(rows)}
    for cohort in ("A", "B"):
        scopes[f"cohort_{cohort}"] = metrics(subset(rows, cohort=cohort))
    for label in DIST_LABELS:
        scopes[f"start_distance|{label}"] = metrics(subset(rows, dist_bucket=label))
    for name, sub in validity_subsets(rows).items():
        scopes[f"validity|{name}"] = metrics(sub)
    per_map = {m: metrics([r for r in rows if r["map"] == m])
               for m in sorted({r["map"] for r in rows})}

    summary = {"split": "test_unseen", "episodes": len(rows),
               "checkpoint_sha": manifest["checkpoint_sha"],
               "manifest_sha": (res / "manifest.sha256").read_text(encoding="utf-8").strip(),
               "step_cap": step_cap, "scopes": scopes, "per_map": per_map,
               "failures": scopes["all"]["failures"],
               "note": "no 03B teacher labels for this split: no length groups, no SPL"}
    (res / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with (res / "summary_tables.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "n", "sr", "ci_lo", "ci_hi", "osr", "stop_rate",
                    "premature_rate", "reached_not_stopped_rate", "oob_rate",
                    "ne_median", "stop_error_median", "steps_mean"])
        for name, m in list(scopes.items()) + [(f"map|{k}", v) for k, v in per_map.items()]:
            if not m.get("n"):
                continue
            w.writerow([name, m["n"], f"{m['sr']:.6f}", f"{m['sr_ci95'][0]:.6f}",
                        f"{m['sr_ci95'][1]:.6f}", f"{m['osr']:.6f}", f"{m['stop_rate']:.6f}",
                        f"{m['premature_stop_rate']:.6f}",
                        f"{m['reached_but_not_stopped_rate']:.6f}",
                        f"{m['out_of_bounds_rate']:.6f}",
                        f"{m['ne_median']:.3f}" if m["ne_median"] is not None else "",
                        f"{m['stop_error_median']:.3f}" if m["stop_error_median"] is not None else "",
                        f"{m['steps_mean']:.3f}"])

    m = scopes["all"]
    lo, hi = m["sr_ci95"]
    print(f"test_unseen n={m['n']} SR={m['sr']:.4f} [{lo:.4f},{hi:.4f}] "
          f"OSR={m['osr']:.4f} STOP={m['stop_rate']:.4f} NEmed={m['ne_median']:.1f}")
    print("failures:", json.dumps(m["failures"]))
    for name, v in per_map.items():
        print(f"  map {name:24s} n={v['n']:5d} SR={v['sr']:.3f} OSR={v['osr']:.3f}")
    for label in DIST_LABELS:
        v = scopes[f"start_distance|{label}"]
        print(f"  start {label:9s} n={v['n']:5d} SR={v['sr']:.3f} OOB={v['out_of_bounds_rate']:.3f}")
    validity = {k: {"n": scopes[f"validity|{k}"]["n"], "sr": round(scopes[f"validity|{k}"]["sr"], 4)}
                for k in ("valid_goal_image", "target_valid_ge_0.5",
                          "target_valid_lt_0.5", "goal_image_invalid")
                if scopes[f"validity|{k}"].get("n")}
    print("validity:", json.dumps(validity))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
