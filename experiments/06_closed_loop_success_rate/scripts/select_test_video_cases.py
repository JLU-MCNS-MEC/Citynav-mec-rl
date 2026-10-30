#!/usr/bin/env python3
"""Pick Test-Unseen episodes for the addendum videos (real stored records).

Successes: one per map (six maps), preferring episodes with turns so the video
is not a straight line.  Failures: premature STOP, the zero-belief STOP on
cambridge_block_32, out-of-bounds, reached-but-not-stopped and the single
collision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results/test_unseen"
OUT = HERE.parent / "videos/test_unseen"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT / "cases.json"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in
            (RES / "journal.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks = {t["episode_id"]: t for t in
             json.loads((RES / "manifest.json").read_text(encoding="utf-8"))["tasks"]}
    cases, used = [], set()

    def turns(r):
        return r["action_counts"]["TurnLeft"] + r["action_counts"]["TurnRight"]

    def add(r, kind, why, label=None):
        used.add(r["episode_id"])
        cases.append({"episode_id": r["episode_id"], "kind": kind,
                      **({"label": label} if label else {}),
                      "split": "test_unseen", "length_group": None,
                      "map": r["map"], "failure": r["failure"], "steps": r["steps"],
                      "why": why})

    maps = sorted({r["map"] for r in rows})
    for name in maps:
        pool = sorted([r for r in rows if r["success"] and r["map"] == name],
                      key=lambda r: (-turns(r), r["episode_id"]))
        if pool:
            add(pool[0], "success", "success on this unseen map; most turns")
        else:
            print(f"  [no success] {name}")

    def first(pred, label, why):
        cand = [r for r in rows if r["episode_id"] not in used and pred(r)]
        if not cand:
            print(f"  [missing category] {label}")
            return
        cand.sort(key=lambda r: r["episode_id"])
        add(cand[0], "failure", why, label)

    first(lambda r: r["failure"] == "premature_stop" and r["map"] != "cambridge_block_32"
          and (r["stop_dist"] or 0) <= 30 and r["steps"] > 20,
          "premature_stop", "stopped 20-30 m from the target")
    first(lambda r: r["map"] == "cambridge_block_32" and r["failure"] == "premature_stop"
          and r["steps"] <= 4,
          "zero_belief_stop", "no candidates on this map -> zero belief -> stops in ~2 steps")
    first(lambda r: r["failure"] == "out_of_bounds" and r["steps"] >= 40,
          "out_of_bounds", "flew out of the map")
    first(lambda r: r["failure"] == "reached_but_not_stopped",
          "reached_but_not_stopped", "entered 20 m and kept flying")
    first(lambda r: r["failure"] == "collision", "collision", "forward move hit a 50 m obstacle")

    payload = {"generated_from": str(RES / "journal.jsonl"),
               "n_success": sum(1 for c in cases if c["kind"] == "success"),
               "n_failure": sum(1 for c in cases if c["kind"] == "failure"),
               "cases": cases}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for c in cases:
        print(f"{c['kind']:7s} {c['episode_id']:16s} {c['map']:22s} "
              f"steps={c['steps']:3d} {c['why']}")
    print(f"wrote {out} ({payload['n_success']} success + {payload['n_failure']} failure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
