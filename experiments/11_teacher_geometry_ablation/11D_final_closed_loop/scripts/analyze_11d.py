#!/usr/bin/env python3
"""11D: merge sharded closed-loop journals and compute SR statistics.

Reads all 11d_loop_* result dirs (main + shards), merges journals per model
+ split, and prints success rates overall / per split / per map / per failure
category, plus stop-distance stats.  Outputs a merged comparison JSON.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path("/mnt/hdd2t/workspaces/Citynav-mec-rl-fixed50m")
HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
BASELINES = {
    "C1 (09C-1)": 0.647,          # Stage 09C-1 closed loop, historical
    "M-v3 (10E)": 0.554,          # Stage 10E, historical
}


def load_recs(out: Path) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    for jf in sorted(out.glob("journal*.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs[rec["episode_id"]] = rec
    return recs


def sr_stats(recs: dict[str, dict], label: str) -> dict:
    n = len(recs)
    if not n:
        return {"label": label, "n": 0}
    no_cand = [r for r in recs.values() if r.get("failure") == "c1_no_candidates"]
    evalable = {k: r for k, r in recs.items()
                if r.get("failure") != "c1_no_candidates"}
    n_ev = len(evalable)
    succ = [r for r in evalable.values() if r["success"]]
    by_split = {}
    for split in ("val_seen", "val_unseen", "test_unseen"):
        s = [r for r in evalable.values() if r["split"] == split]
        if s:
            by_split[split] = {"n": len(s),
                               "sr": sum(r["success"] for r in s) / len(s),
                               "osr": sum(r["osr"] for r in s) / len(s)}
    by_map = {}
    for name in sorted({r["map"] for r in evalable.values()}):
        s = [r for r in evalable.values() if r["map"] == name]
        by_map[name] = {"n": len(s),
                        "sr": sum(r["success"] for r in s) / len(s)}
    failures = Counter(r["failure"] for r in evalable.values() if not r["success"])
    return {"label": label, "n": n, "n_evalable": n_ev,
            "n_no_candidates": len(no_cand),
            "sr": sum(r["success"] for r in evalable.values()) / n_ev if n_ev else 0.0,
            "by_split": by_split, "by_map": by_map,
            "failures": dict(failures),
            "mean_steps": sum(r["steps"] for r in evalable.values()) / n_ev if n_ev else 0.0,
            "stop_rate": sum(r["stopped"] for r in evalable.values()) / n_ev if n_ev else 0.0,
            "stop_dist_mean": (sum(r["stop_dist"] for r in evalable.values()
                                   if r["stop_dist"] is not None)
                               / max(1, sum(1 for r in evalable.values()
                                            if r["stop_dist"] is not None)))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["v4", "v5"])
    parser.add_argument("--prefix", default=None,
                        help="e.g. 11d_loop_v4 (defaults to 11d_loop_<model>)")
    args = parser.parse_args()
    prefix = args.prefix or f"11d_loop_{args.model}"

    val = {}
    for d in sorted(RESULTS.glob(f"{prefix}_val*")):
        val.update(load_recs(d))
    test = {}
    for d in sorted(RESULTS.glob(f"{prefix}_test*")):
        test.update(load_recs(d))
    all_recs = {**val, **test}
    out = {"model": args.model, "prefix": prefix,
           "val": sr_stats(val, "val_seen+val_unseen"),
           "test": sr_stats(test, "test_unseen"),
           "all": sr_stats(all_recs, "all"),
           "baselines": BASELINES}
    print(json.dumps({"val": out["val"], "test": out["test"],
                      "all": {"n": out["all"]["n"], "sr": out["all"]["sr"],
                              "by_split": out["all"]["by_split"]}},
                     indent=1, ensure_ascii=False))
    (RESULTS / f"{prefix}_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
