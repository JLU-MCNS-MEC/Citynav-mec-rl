#!/usr/bin/env python3
"""09B paired analysis: dynamic belief vs Transformer-M / static Top-1 / Top-5.

Reads the existing Stage-06 and Stage-08 journals (never recomputed) plus the
09B journal.  Produces comparison JSON/CSV and the update-quality statistics
(did an update move the MAP closer to the true target?).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RES = HERE.parent / "results"
sys.path.insert(0, str(ROOT / "06_closed_loop_success_rate/scripts"))
from summarize_results import metrics  # noqa: E402

STAGE06 = ROOT / "06_closed_loop_success_rate/results/seed42"
STAGE08 = ROOT / "08_belief_only_geometric_navigation/results"
DYN = RES / "09b_dynamic"
OUT = RES


def load(path: Path) -> dict[str, dict]:
    return {json.loads(l)["episode_id"]: json.loads(l)
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def paired(a: dict, b: dict, ids: list[str]) -> dict:
    both = only_a = only_b = neither = 0
    for i in ids:
        sa, sb = bool(a[i]["success"]), bool(b[i]["success"])
        both += sa and sb
        only_a += sa and not sb
        only_b += (not sa) and sb
        neither += (not sa) and (not sb)
    n = len(ids)
    diff = (only_b - only_a) / n
    se = math.sqrt(max(only_a + only_b - (only_b - only_a) ** 2 / n, 0)) / n
    try:
        from scipy.stats import binomtest
        p = float(binomtest(only_b, only_a + only_b, 0.5).pvalue) if only_a + only_b else 1.0
    except Exception:                                   # noqa: BLE001
        p = None
    return {"n": n, "both": both, "only_a": only_a, "only_b": only_b, "neither": neither,
            "sr_a": (both + only_a) / n, "sr_b": (both + only_b) / n,
            "diff_b_minus_a": diff, "ci95": [diff - 1.96 * se, diff + 1.96 * se],
            "mcnemar_p": p}


def main() -> int:
    t = load(STAGE06 / "journal.jsonl")
    g1 = load(STAGE08 / "top1/journal.jsonl")
    g5 = load(STAGE08 / "top5/journal.jsonl")
    d = load(DYN / "journal.jsonl")
    ids = sorted(set(t) & set(g1) & set(g5) & set(d))
    methods = {"transformer": t, "top1_static": g1, "top5_static": g5, "dynamic": d}
    manifest = {x["episode_id"]: x for x in
                json.loads((STAGE06 / "manifest.json").read_text(encoding="utf-8"))["tasks"]}

    out = {"episodes": len(ids), "methods": {}, "paired": {}, "by_group": {},
           "updates": {}}
    for name, rows in methods.items():
        out["methods"][name] = metrics([rows[i] for i in ids])
    for label, (a, b) in {"dynamic_vs_transformer": ("transformer", "dynamic"),
                          "dynamic_vs_top1": ("top1_static", "dynamic"),
                          "top1_vs_transformer": ("transformer", "top1_static")}.items():
        out["paired"][label] = paired(methods[a], methods[b], ids)
    for split in ("val_seen", "val_unseen"):
        for g in ("short", "medium", "long"):
            sub = [i for i in ids if manifest[i]["split"] == split
                   and manifest[i]["length_group"] == g]
            out["by_group"][f"{split}|{g}"] = {
                name: metrics([rows[i] for i in sub]) for name, rows in methods.items()}
            out["by_group"][f"{split}|{g}"]["paired_vs_top1"] = paired(g1, d, sub)

    # update quality: does the MAP get closer to the true target after an update?
    improve = worsen = same = 0
    pre_err, post_err, vis_err = [], [], []
    for eid in ids:
        path = DYN / "episodes" / f"{eid.replace(':', '_')}.json"
        rec = json.loads(path.read_text(encoding="utf-8"))
        target = np.asarray(rec["target"], float)
        prev_err = None
        for s in rec["trajectory"]:
            map_pos = np.asarray(s["map_candidate"], float)
            err = float(np.linalg.norm(map_pos - target))
            vu = s.get("visual_offset_m")
            if s.get("confident") and vu is not None:
                vis_pos = np.array([s["x"] + vu[0], s["y"] + vu[1]])
                vis_err.append(float(np.linalg.norm(vis_pos - target)))
            if s.get("updated") and prev_err is not None:
                if err < prev_err - 1e-6:
                    improve += 1
                elif err > prev_err + 1e-6:
                    worsen += 1
                else:
                    same += 1
                pre_err.append(prev_err)
                post_err.append(err)
            prev_err = err
    out["updates"] = {"n": improve + worsen + same, "improved": improve,
                      "worsened": worsen, "unchanged": same,
                      "pre_median_m": float(np.median(pre_err)) if pre_err else None,
                      "post_median_m": float(np.median(post_err)) if post_err else None,
                      "visual_match_median_m": float(np.median(vis_err)) if vis_err else None}

    (OUT / "09b_comparison.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    with (OUT / "09b_comparison_tables.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n", "sr", "ci_lo", "ci_hi", "osr", "stop_rate",
                    "premature_rate", "reached_rate", "oob_rate", "ne_median", "steps_mean"])
        for name, m in out["methods"].items():
            w.writerow([name, m["n"], f"{m['sr']:.6f}", f"{m['sr_ci95'][0]:.6f}",
                        f"{m['sr_ci95'][1]:.6f}", f"{m['osr']:.6f}", f"{m['stop_rate']:.6f}",
                        f"{m['premature_stop_rate']:.6f}",
                        f"{m['reached_but_not_stopped_rate']:.6f}",
                        f"{m['out_of_bounds_rate']:.6f}", f"{m['ne_median']:.3f}",
                        f"{m['steps_mean']:.2f}"])
        w.writerow([])
        w.writerow(["group", "method", "n", "sr", "osr", "oob_rate", "steps_mean"])
        for key, block in out["by_group"].items():
            for name in methods:
                m = block[name]
                w.writerow([key, name, m["n"], f"{m['sr']:.6f}", f"{m['osr']:.6f}",
                            f"{m['out_of_bounds_rate']:.6f}", f"{m['steps_mean']:.2f}"])

    print(json.dumps({k: {kk: out["methods"][k][kk] for kk in
                          ("n", "sr", "sr_ci95", "osr", "stop_rate",
                           "out_of_bounds_rate", "ne_median", "steps_mean")}
                      for k in methods}, indent=1))
    print(json.dumps(out["paired"], indent=1))
    print("updates:", json.dumps(out["updates"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
