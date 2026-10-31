#!/usr/bin/env python3
"""09C analysis: continuous localisation, gating and paired SR comparisons.

Compares six methods on the same 5,324 tasks and measures:
* SR/OSR/CI, paired differences and McNemar;
* continuous estimate error at visible states, sub-grid offset preservation;
* accepted-match correctness, out-of-view high-confidence rate;
* update quality (improved / worsened).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RES = HERE.parent / "results"
sys.path.insert(0, str(ROOT / "experiments/06_closed_loop_success_rate/scripts"))
from summarize_results import metrics  # noqa: E402

STAGE06 = ROOT / "experiments/06_closed_loop_success_rate/results/seed42"
STAGE08 = ROOT / "experiments/08_belief_only_geometric_navigation/results"
STAGE09 = ROOT / "experiments/09_online_visual_goal_matching/results"
METHODS = {"transformer": STAGE06 / "journal.jsonl",
           "top1_static": STAGE08 / "top1/journal.jsonl",
           "top5_static": STAGE08 / "top5/journal.jsonl",
           "dynamic_24m": STAGE09 / "09b_dynamic/journal.jsonl",
           "continuous_c1": RES / "09c_c1/journal.jsonl",
           "continuous_c2": RES / "09c_c2/journal.jsonl"}


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
            "diff": diff, "ci95": [diff - 1.96 * se, diff + 1.96 * se], "mcnemar_p": p}


def visible(x, y, yaw, tx, ty) -> tuple[bool, float]:
    dx, dy = tx - x, ty - y
    d = math.hypot(dx, dy)
    if d > 48.0:
        return False, d
    f = dx * math.cos(yaw) + dy * math.sin(yaw)
    l = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return (abs(f) <= 50 and abs(l) <= 50), d


def main() -> int:
    journals = {k: load(v) for k, v in METHODS.items()}
    ids = sorted(set.intersection(*(set(j) for j in journals.values())))
    print("common episodes:", len(ids))
    manifest = {t["episode_id"]: t for t in
                json.loads((STAGE06 / "manifest.json").read_text(encoding="utf-8"))["tasks"]}
    out = {"episodes": len(ids), "methods": {}, "paired": {}, "by_group": {},
           "localisation": {}, "gating": {}}

    for name, rows in journals.items():
        out["methods"][name] = metrics([rows[i] for i in ids])
    for label, (a, b) in {"c1_vs_top1": ("top1_static", "continuous_c1"),
                          "c2_vs_top1": ("top1_static", "continuous_c2"),
                          "c2_vs_c1": ("continuous_c1", "continuous_c2"),
                          "c2_vs_dynamic24": ("dynamic_24m", "continuous_c2"),
                          "c2_vs_transformer": ("transformer", "continuous_c2")}.items():
        out["paired"][label] = paired(journals[a], journals[b], ids)
    for split in ("val_seen", "val_unseen"):
        for g in ("short", "medium", "long"):
            sub = [i for i in ids if manifest[i]["split"] == split
                   and manifest[i]["length_group"] == g]
            out["by_group"][f"{split}|{g}"] = {
                k: metrics([rows[i] for i in sub]) for k, rows in journals.items()}

    # localisation / gating statistics per continuous mode
    for mode in ("c1", "c2"):
        folder = RES / f"09c_{mode}" / "episodes"
        est_err, vis_err, subgrid_better, subgrid_n = [], [], 0, 0
        accepted = correct_accept = wrong_accept = 0
        oov_steps = oov_conf = 0
        improved = worsened = 0
        for eid in ids:
            rec = json.loads((folder / f"{eid.replace(':', '_')}.json").read_text(encoding="utf-8"))
            t = np.asarray(rec["target"])
            prev = None
            for s in rec["trajectory"]:
                vis, dist = visible(s["x"], s["y"], s["yaw"], t[0], t[1])
                est = np.asarray(s["estimate"])
                ev = float(np.linalg.norm(est - t))
                ve = s.get("visual_err_to_true_m")
                if vis:
                    est_err.append(ev)
                    if ve is not None:
                        vis_err.append(ve)
                if s["accept"]:
                    accepted += 1
                    if ve is not None and ve <= 10:
                        correct_accept += 1
                    elif ve is not None and ve > 20:
                        wrong_accept += 1
                if not vis:
                    oov_steps += 1
                    if s["accept"]:
                        oov_conf += 1
                if prev is not None and s.get("updated"):
                    if ev < prev - 1e-9:
                        improved += 1
                    elif ev > prev + 1e-9:
                        worsened += 1
                prev = ev
        out["localisation"][mode] = {
            "est_median_m": float(np.median(est_err)) if est_err else None,
            "est_mae_m": float(np.mean(est_err)) if est_err else None,
            "est_p90_m": float(np.percentile(est_err, 90)) if est_err else None,
            "est_prec_5": float(np.mean(np.array(est_err) <= 5)) if est_err else None,
            "est_prec_10": float(np.mean(np.array(est_err) <= 10)) if est_err else None,
            "est_prec_20": float(np.mean(np.array(est_err) <= 20)) if est_err else None,
            "visual_median_m": float(np.median(vis_err)) if vis_err else None,
        }
        out["gating"][mode] = {
            "accepted_steps": accepted, "correct_accept_10m": correct_accept,
            "wrong_accept_20m": wrong_accept,
            "accept_precision_10m": correct_accept / max(accepted, 1),
            "wrong_accept_rate": wrong_accept / max(accepted, 1),
            "out_of_view_steps": oov_steps,
            "out_of_view_accepted": oov_conf,
            "out_of_view_accept_rate": oov_conf / max(oov_steps, 1),
            "updates_improved": improved, "updates_worsened": worsened,
            "update_precision": improved / max(improved + worsened, 1),
        }

    (RES / "09c_comparison.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    with (RES / "09c_comparison_tables.csv").open("w", newline="", encoding="utf-8") as fh:
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
        w.writerow(["comparison", "n", "both", "only_a", "only_b", "neither",
                    "diff", "ci_lo", "ci_hi", "mcnemar_p"])
        for label, p in out["paired"].items():
            w.writerow([label, p["n"], p["both"], p["only_a"], p["only_b"], p["neither"],
                        f"{p['diff']:.6f}", f"{p['ci95'][0]:.6f}", f"{p['ci95'][1]:.6f}",
                        f"{p['mcnemar_p']:.3g}"])
    print(json.dumps({k: {kk: out["methods"][k][kk] for kk in
                          ("n", "sr", "sr_ci95", "osr", "premature_stop_rate",
                           "out_of_bounds_rate", "ne_median", "steps_mean")}
                      for k in METHODS}, indent=1))
    print(json.dumps(out["paired"], indent=1))
    print("localisation:", json.dumps(out["localisation"], indent=1))
    print("gating:", json.dumps(out["gating"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
