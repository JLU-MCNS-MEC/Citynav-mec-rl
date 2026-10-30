#!/usr/bin/env python3
"""Pick the real Stage-06 episodes that the videos replay.

Successes: two per (split, length group), chosen for map diversity and turn
complexity (a plain straight line is not representative).  Failures: one per
diagnostic category, using only categories that actually occur in the journal.

Writes ``videos/seed42/cases.json`` and prints the selection with its rationale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results/seed42"
OUT = HERE.parent / "videos/seed42"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT / "cases.json"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in
            (RES / "journal.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {r["episode_id"]: r for r in rows}
    succ = [r for r in rows if r["success"]]

    cases = []

    def turns(r):
        return r["action_counts"]["TurnLeft"] + r["action_counts"]["TurnRight"]

    used_ids, used_maps = set(), set()
    for split in ("val_seen", "val_unseen"):
        for group in ("short", "medium", "long"):
            pool = [r for r in succ if r["split"] == split and r["length_group"] == group]
            # prefer more turns, then map diversity, then the smaller episode id
            pool.sort(key=lambda r: (-turns(r), r["episode_id"]))
            picked = 0
            for prefer_new_map in (True, False):
                for r in pool:
                    if picked >= 2:
                        break
                    if r["episode_id"] in used_ids:
                        continue
                    if prefer_new_map and r["map"] in used_maps:
                        continue
                    used_ids.add(r["episode_id"])
                    used_maps.add(r["map"])
                    cases.append({"episode_id": r["episode_id"], "kind": "success",
                                  "split": split, "length_group": group,
                                  "map": r["map"], "turns": turns(r),
                                  "steps": r["steps"],
                                  "why": "success; map-diverse, turn-complex"})
                    picked += 1

    def first(pred, label, why):
        cand = [r for r in rows if pred(r) and r["episode_id"] not in used_ids]
        if not cand:
            print(f"  [missing category] {label}")
            return
        cand.sort(key=lambda r: r["episode_id"])
        r = cand[0]
        used_ids.add(r["episode_id"])
        cases.append({"episode_id": r["episode_id"], "kind": "failure", "label": label,
                      "split": r["split"], "length_group": r["length_group"],
                      "map": r["map"], "failure": r["failure"], "steps": r["steps"],
                      "why": why})

    first(lambda r: r["failure"] == "premature_stop" and 20 < (r["stop_dist"] or 0) <= 30
          and turns(r) >= 1,
          "premature_stop", "stopped 20-30 m from the target")
    first(lambda r: r["failure"] == "reached_but_not_stopped" and r["min_dist"] is not None
          and r["min_dist"] <= 15,
          "reached_but_not_stopped", "entered 15 m, kept flying")
    first(lambda r: r["failure"] == "out_of_bounds" and r["length_group"] == "long"
          and turns(r) == 0,
          "long_straight_oob", "long task, zero turns, flew out of the map")
    first(lambda r: r["failure"] in ("out_of_bounds", "premature_stop")
          and turns(r) >= 4 and r["steps"] >= 20,
          "turn_oscillation", "many turns without progress")
    first(lambda r: r.get("target_valid_fraction") is not None
          and 0 < r["target_valid_fraction"] < 0.5,
          "low_valid_target", "target patch < 50% valid pixels")
    first(lambda r: r["failure"] == "collision", "collision", "forward move hit a 50 m obstacle")

    payload = {"generated_from": str(RES / "journal.jsonl"),
               "n_success": sum(1 for c in cases if c["kind"] == "success"),
               "n_failure": sum(1 for c in cases if c["kind"] == "failure"),
               "cases": cases}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for c in cases:
        print(f"{c['kind']:7s} {c['episode_id']:16s} {c['split']:10s} "
              f"{c['length_group']:6s} {c['map']:22s} steps={c['steps']:3d} {c['why']}")
    print(f"wrote {out} ({payload['n_success']} success + {payload['n_failure']} failure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
