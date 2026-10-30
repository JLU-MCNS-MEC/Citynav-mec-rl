#!/usr/bin/env python3
"""Paired analysis: Transformer-M vs Top-1 / Top-5 geometric controllers.

Reads the Stage-06 journal (reused, never recomputed) and the two geometry
journals.  Produces comparison.json, comparison_tables.csv, paired_outcomes.csv
and figures/comparison.png.

The paired difference uses discordant pairs (b = only A succeeds, c = only B):
  diff = (b - c) / n,  SE = sqrt(b + c - (b - c)^2 / n) / n
and an exact two-sided McNemar test (binomial on b, b + c).
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
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "06_closed_loop_success_rate/scripts"))

from summarize_results import metrics  # noqa: E402

STAGE06 = ROOT / "06_closed_loop_success_rate/results/seed42"
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"
GROUPS = ("short", "medium", "long")
SPLITS = ("val_seen", "val_unseen")


def load(path: Path) -> dict[str, dict]:
    return {json.loads(line)["episode_id"]: json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def paired(a: dict, b: dict, ids: list[str]) -> dict:
    both = only_a = only_b = neither = 0
    for i in ids:
        sa, sb = bool(a[i]["success"]), bool(b[i]["success"])
        if sa and sb:
            both += 1
        elif sa:
            only_a += 1
        elif sb:
            only_b += 1
        else:
            neither += 1
    n = len(ids)
    diff = (only_b - only_a) / n
    disc = only_a + only_b
    se = math.sqrt(max(disc - (only_b - only_a) ** 2 / n, 0)) / n if n else 0.0
    try:
        from scipy.stats import binomtest
        p = float(binomtest(only_b, disc, 0.5).pvalue) if disc else 1.0
    except Exception:                                     # noqa: BLE001
        p = None
    return {"n": n, "both": both, "only_a": only_a, "only_b": only_b, "neither": neither,
            "sr_a": (both + only_a) / n, "sr_b": (both + only_b) / n,
            "diff_b_minus_a": diff, "diff_ci95": [diff - 1.96 * se, diff + 1.96 * se],
            "mcnemar_p": p}


EPISODE_DIRS = {
    "transformer": STAGE06 / "episodes",
    "top1": RES / "top1/episodes",
    "top5": RES / "top5/episodes",
}


def bearing_stats(rows: dict, manifest: dict, method: str) -> dict:
    """Fraction of steps with the true target behind that chose a turn."""
    behind = turn_behind = 0
    turn_steps = steps = 0
    oob = 0
    for eid, rec in rows.items():
        t = manifest[eid]
        path = EPISODE_DIRS[method] / f"{eid.replace(':', '_')}.json"
        st = json.loads(path.read_text(encoding="utf-8")).get("trajectory") or []
        oob += int(bool(rec.get("out_of_bounds")))
        for s in st:
            steps += 1
            b = abs(math.atan2(t["target_y"] - s["y"], t["target_x"] - s["x"]) - s["yaw"])
            b = (b + math.pi) % (2 * math.pi) - math.pi
            is_turn = s["action"] in (2, 3)
            turn_steps += int(is_turn)
            if abs(b) > math.pi / 2:
                behind += 1
                turn_behind += int(is_turn)
    return {"steps": steps, "turn_share": turn_steps / max(steps, 1),
            "behind_states": behind,
            "turn_rate_when_behind": turn_behind / max(behind, 1),
            "oob_count": oob}


def main() -> int:
    manifest = {t["episode_id"]: t for t in
                json.loads((STAGE06 / "manifest.json").read_text(encoding="utf-8"))["tasks"]}
    t = load(STAGE06 / "journal.jsonl")
    g1 = load(RES / "top1/journal.jsonl")
    g5 = load(RES / "top5/journal.jsonl")
    ids = sorted(set(t) & set(g1) & set(g5))
    assert len(ids) == 5324, len(ids)

    methods = {"transformer": t, "top1": g1, "top5": g5}
    out = {"episodes": len(ids), "methods": {}, "paired": {}, "by_split": {},
           "by_group": {}, "behaviour": {}}
    for name, rows in methods.items():
        sel = [rows[i] for i in ids]
        out["methods"][name] = metrics(sel)

    for label, (a, b) in {"transformer_vs_top1": ("transformer", "top1"),
                          "transformer_vs_top5": ("transformer", "top5"),
                          "top1_vs_top5": ("top1", "top5")}.items():
        out["paired"][label] = paired(methods[a], methods[b], ids)

    for split in SPLITS:
        sids = [i for i in ids if manifest[i]["split"] == split]
        out["by_split"][split] = {
            name: metrics([rows[i] for i in sids]) for name, rows in methods.items()}
        out["by_split"][split]["paired_t1"] = paired(t, g1, sids)
        out["by_split"][split]["paired_t5"] = paired(t, g5, sids)
        for group in GROUPS:
            gids = [i for i in sids if manifest[i]["length_group"] == group]
            out["by_group"][f"{split}|{group}"] = {
                name: metrics([rows[i] for i in gids]) for name, rows in methods.items()}
            out["by_group"][f"{split}|{group}"]["paired_t1"] = paired(t, g1, gids)
            out["by_group"][f"{split}|{group}"]["paired_t5"] = paired(t, g5, gids)

    for name, rows in methods.items():
        out["behaviour"][name] = bearing_stats(rows, manifest, name)
        count = Counter(r["failure"] for r in rows.values())
        out["behaviour"][name]["failures"] = dict(count)

    # STOP quality: estimated distance at the STOP step vs the true distance
    for name in ("top1", "top5"):
        diffs, premature_diffs, wrong = [], [], 0
        for eid, r in methods[name].items():
            if r.get("stop_dist") is None:
                continue
            path = RES / name / "episodes" / f"{eid.replace(':', '_')}.json"
            rec = json.loads(path.read_text(encoding="utf-8"))
            est = (rec.get("trajectory") or [{}])[-1].get("est_distance_m")
            if est is None:
                continue
            diffs.append(est - r["stop_dist"])
            if r["failure"] == "premature_stop":
                premature_diffs.append(est - r["stop_dist"])
                wrong += int(est <= 20 < r["stop_dist"])
        out["behaviour"][name]["stop_estimation"] = {
            "n_stopped": len(diffs),
            "signed_error_mean": float(np.mean(diffs)) if diffs else None,
            "abs_error_median": float(np.median(np.abs(diffs))) if diffs else None,
            "premature_n": len(premature_diffs),
            "premature_est_median": float(np.median(premature_diffs)) if premature_diffs else None,
            "wrong_stop_from_underestimate": wrong}

    # cases where Transformer succeeds but geometry fails (and vice versa)
    t_success = {i for i in ids if t[i]["success"]}
    g1_success = {i for i in ids if g1[i]["success"]}
    g5_success = {i for i in ids if g5[i]["success"]}
    out["case_sets"] = {
        "t_ok_g1_fail": sorted(t_success - g1_success),
        "t_fail_g1_ok": sorted(g1_success - t_success),
        "t_ok_g5_fail": sorted(t_success - g5_success),
        "t_fail_g5_ok": sorted(g5_success - t_success),
    }

    (RES / "comparison.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    with (RES / "comparison_tables.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "method", "n", "sr", "ci_lo", "ci_hi", "osr", "stop_rate",
                    "premature_rate", "reached_rate", "oob_rate", "ne_median", "steps_mean"])
        for scope, block in (("all", {"m": out["methods"]}),
                             *[(f"{s}", out["by_split"][s]) for s in SPLITS]):
            if scope == "all":
                table = out["methods"]
            else:
                table = {k: v for k, v in block.items() if isinstance(v, dict) and "sr" in v}
            for name, m in table.items():
                w.writerow([scope, name, m["n"], f"{m['sr']:.6f}", f"{m['sr_ci95'][0]:.6f}",
                            f"{m['sr_ci95'][1]:.6f}", f"{m['osr']:.6f}", f"{m['stop_rate']:.6f}",
                            f"{m['premature_stop_rate']:.6f}",
                            f"{m['reached_but_not_stopped_rate']:.6f}",
                            f"{m['out_of_bounds_rate']:.6f}", f"{m['ne_median']:.3f}",
                            f"{m['steps_mean']:.2f}"])
        rows = []
        for key, block in out["by_group"].items():
            for name in methods:
                m = block[name]
                rows.append([key, name, m["n"], f"{m['sr']:.6f}", f"{m['osr']:.6f}",
                             f"{m['out_of_bounds_rate']:.6f}", f"{m['steps_mean']:.2f}"])
        w.writerow([])
        w.writerow(["split|group", "method", "n", "sr", "osr", "oob_rate", "steps_mean"])
        for r in rows:
            w.writerow(r)

    with (RES / "paired_outcomes.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["comparison", "n", "both", "only_a", "only_b", "neither",
                    "sr_a", "sr_b", "diff_b_minus_a", "ci_lo", "ci_hi", "mcnemar_p"])
        for label, p in out["paired"].items():
            w.writerow([label, p["n"], p["both"], p["only_a"], p["only_b"], p["neither"],
                        f"{p['sr_a']:.6f}", f"{p['sr_b']:.6f}", f"{p['diff_b_minus_a']:.6f}",
                        f"{p['diff_ci95'][0]:.6f}", f"{p['diff_ci95'][1]:.6f}",
                        f"{p['mcnemar_p']:.3g}"])

    print(json.dumps({k: {kk: out["methods"][k][kk] for kk in
                          ("n", "sr", "sr_ci95", "osr", "stop_rate", "out_of_bounds_rate",
                           "ne_median", "steps_mean")} for k in methods}, indent=1))
    for label, p in out["paired"].items():
        print(label, p)
    print("behaviour:", json.dumps(out["behaviour"], indent=1)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
