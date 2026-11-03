#!/usr/bin/env python3
"""Case selection for V4/V5/V3 closed-loop comparison videos.

Categories (strictly paired by episode_id; same map/start/target verified
from the episode JSONs, never inferred from the CSV alone):
  A: V5 success / V4 failure   (from per_episode_alignment.csv)
  B: V4 success / V3 failure   (from v4_v3_alignment.csv, corrected-protocol
                                V3 re-eval, val only)
  C: V4 success / V5 failure   (from per_episode_alignment.csv)

Ranking is deterministic (seed 0 = manifest order), greedy with diversity:
  * unique map per category
  * loser failure-reason diversity (relaxed only if the pool is too small)
  * belief story: winner final est_distance <= 8 m AND final estimate within
    20 m of target; loser belief drift (max est_distance >= 30 m)
  * steps: loser steps in [8,160], winner not stopped by step cap
  * split mix: test_unseen preferred for A/C (no test case videos exist yet)
Excluded by default: cambridge_block_32 (empty candidates -- belief story is
impossible there) and collision losers for A (only 2 in the whole test set).
Overridable per episode via --override (membership still validated).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
assert (REPO / "experiments").is_dir(), f"REPO 解析错误: {REPO}"
OUT = REPO / "reports/v4_v5_comparison/case_studies"
ALIGN_V4V5 = REPO / "reports/v4_v5_comparison/per_episode_alignment.csv"
ALIGN_V4V3 = REPO / "reports/v4_v5_comparison/v4_v3_alignment.csv"
EPISODE_ROOTS = {
    "v4": [REPO / "experiments/v4_closed_loop_revalidation/results"],
    "v5": [REPO / "experiments/v5_closed_loop_evaluation/results"],
    "v3": [REPO / "experiments/v3_closed_loop_revalidation/results"],
}
EPISODE_PREFIXES = {"v4": "full_", "v5": "v5_", "v3": "v3_"}
CATEGORIES = {
    "A": {"winner": "v5", "loser": "v4", "csv": ALIGN_V4V5},
    "B": {"winner": "v4", "loser": "v3", "csv": ALIGN_V4V3},
    "C": {"winner": "v4", "loser": "v5", "csv": ALIGN_V4V5},
}
PREFERRED_FAILURES = {"A": ["out_of_bounds", "reached_but_not_stopped",
                            "never_stop"],
                      "B": ["out_of_bounds", "reached_but_not_stopped",
                            "never_stop", "premature_stop"],
                      "C": ["out_of_bounds", "premature_stop",
                            "reached_but_not_stopped", "never_stop"]}
EXCLUDE_MAPS = {"cambridge_block_32"}          # empty candidates
EXCLUDE_FAILURES = {"A": {"collision"}}        # 2 cases in the whole set
SPLIT_PREF = {"test_unseen": 1.0, "val_unseen": 0.5, "val_seen": 0.0,
              None: 0.0}
STOP_RADIUS = 20.0


# ---------------------------------------------------------------- data layer


def load_alignment_csv(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            row = dict(r)
            for k, v in row.items():
                if k in ("episode_id", "split", "map"):
                    continue
                if v == "True":
                    row[k] = True
                elif v == "False":
                    row[k] = False
                elif v in ("", "None"):
                    row[k] = None
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        pass
            rows[row["episode_id"]] = row
    return rows


def load_episode_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def locate_episode(model: str, eid: str,
                   roots: dict[str, list[Path]] | None = None) -> Path | None:
    roots = roots or EPISODE_ROOTS
    fn = f"{eid.replace(':', '_')}.json"
    for d in sorted(roots[model]):
        for ed in sorted(d.glob(f"{EPISODE_PREFIXES[model]}*/episodes")):
            p = ed / fn
            if p.exists():
                return p
    return None


# --------------------------------------------------------------- belief stats


def belief_stats(rec: dict) -> dict:
    """Extract the real per-step belief numbers the ranking (and the video
    metadata) are built from. Never fabricated: absent trajectory -> None."""
    traj = rec.get("trajectory") or []
    if not traj:
        return {"final_est_m": None, "est_to_target_m": None,
                "max_est_m": None, "story_ok": False, "drift_ok": False}
    ests = [s.get("est_distance_m") for s in traj
            if s.get("est_distance_m") is not None]
    max_est = max(ests) if ests else None
    last = traj[-1]
    final_est = last.get("est_distance_m")
    target = rec.get("target")
    est_to_target = None
    if last.get("estimate") is not None and target is not None:
        ex, ey = last["estimate"][0], last["estimate"][1]
        est_to_target = float(((ex - target[0]) ** 2
                               + (ey - target[1]) ** 2) ** 0.5)
    story_ok = (final_est is not None and est_to_target is not None
                and final_est <= 8.0 and est_to_target <= 20.0)
    drift_ok = max_est is not None and max_est >= 30.0
    return {"final_est_m": final_est, "est_to_target_m": est_to_target,
            "max_est_m": max_est, "story_ok": story_ok,
            "drift_ok": drift_ok}


# --------------------------------------------------------------- candidate set


def candidate_pair(cat: str, eid: str,
                   roots: dict[str, list[Path]] | None = None
                   ) -> tuple[dict | None, dict | None, str]:
    """Load both episodes of a candidate pair and verify strict pairing.

    Returns (winner_rec, loser_rec, err). err values:
      "" ok | "missing" | "task_mismatch" | "winner_not_success" |
      "loser_success" | "winner_stop_cap"
    """
    cfg = CATEGORIES[cat]
    w_path = locate_episode(cfg["winner"], eid, roots)
    l_path = locate_episode(cfg["loser"], eid, roots)
    if w_path is None or l_path is None:
        return None, None, "missing"
    w, l = load_episode_json(w_path), load_episode_json(l_path)
    if (w["map"] != l["map"] or w["start"][:2] != l["start"][:2]
            or w["start"][2] != l["start"][2] or w["target"] != l["target"]):
        return None, None, "task_mismatch"
    if not w["success"] or not (w.get("stopped")
                                and w.get("stop_dist") is not None
                                and w["stop_dist"] <= STOP_RADIUS):
        return None, None, "winner_not_success"
    if l["success"]:
        return None, None, "loser_success"
    if w.get("stop_reason") == "step_cap":
        return None, None, "winner_stop_cap"
    w["_path"], l["_path"] = str(w_path), str(l_path)
    return w, l, ""


def pool_for(cat: str, alignment: dict[str, dict],
             roots: dict[str, list[Path]] | None = None) -> list[str]:
    cfg = CATEGORIES[cat]
    w_col = f"{cfg['winner']}_success"
    l_col = f"{cfg['loser']}_success"
    out = []
    for eid, row in sorted(alignment.items()):
        if row.get(w_col) is True and row.get(l_col) is False:
            out.append(eid)
    return out


# ------------------------------------------------------------------- scoring


def score(cat: str, winner: dict, loser: dict) -> tuple:
    """Lexicographic score, higher = more representative. Deterministic."""
    wb, lb = belief_stats(winner), belief_stats(loser)
    steps_ok = (8 <= loser["steps"] <= 160 and winner["steps"] >= 2)
    failure_pref = loser.get("failure") in PREFERRED_FAILURES[cat]
    split_pref = SPLIT_PREF.get(loser.get("split"))
    return (wb["story_ok"], lb["drift_ok"], steps_ok, failure_pref, split_pref,
            -abs(loser["steps"] - 40))


def greedy_select(cat: str, candidates: list[str], n: int,
                  roots: dict[str, list[Path]] | None = None,
                  exclude_empty_cand: bool = True,
                  seed: int = 0) -> tuple[list[dict], list[str]]:
    """Deterministic greedy selection with map + failure diversity.

    Relaxation order if the strict pass underfills: failure diversity,
    then map uniqueness. Notes record every exclusion and relaxation."""
    notes: list[str] = []
    valid: list[tuple[str, dict, dict]] = []
    for eid in candidates:
        w, l, err = candidate_pair(cat, eid, roots)
        if err:
            notes.append(f"{eid}: excluded ({err})")
            continue
        if exclude_empty_cand and l["map"] in EXCLUDE_MAPS:
            notes.append(f"{eid}: excluded (empty-candidate map {l['map']})")
            continue
        if l.get("failure") in EXCLUDE_FAILURES.get(cat, set()):
            notes.append(f"{eid}: excluded (failure {l['failure']})")
            continue
        valid.append((eid, w, l))
    rng = random.Random(seed)
    ordered = sorted(valid, key=lambda t: score(cat, t[1], t[2]), reverse=True)
    if seed != 0:
        rng.shuffle(ordered)

    picks: list[dict] = []
    used_maps: set[str] = set()
    used_failures: set[str] = set()

    def take(relax_fail: bool, relax_map: bool) -> None:
        for eid, w, l in ordered:
            if len(picks) >= n:
                return
            if any(p["episode_id"] == eid for p in picks):
                continue
            if not relax_map and l["map"] in used_maps:
                continue
            if not relax_fail and l["failure"] in used_failures:
                continue
            wb, lb = belief_stats(w), belief_stats(l)
            picks.append({"category": cat, "episode_id": eid,
                          "split": l["split"], "map": l["map"],
                          "winner_model": CATEGORIES[cat]["winner"],
                          "loser_model": CATEGORIES[cat]["loser"],
                          "winner_ref": w["_path"], "loser_ref": l["_path"],
                          "winner_steps": w["steps"],
                          "loser_steps": l["steps"],
                          "loser_failure": l["failure"],
                          "winner_final_est_m": wb["final_est_m"],
                          "winner_est_to_target_m": wb["est_to_target_m"],
                          "loser_max_est_m": lb["max_est_m"],
                          "rationale": _rationale(cat, w, l, wb, lb)})
            used_maps.add(l["map"])
            used_failures.add(l["failure"])

    take(False, False)
    if len(picks) < n:
        notes.append(f"{cat}: pool too small for full diversity, relaxing "
                     f"failure-reason diversity")
        take(True, False)
    if len(picks) < n:
        notes.append(f"{cat}: pool too small for map uniqueness, relaxing")
        take(True, True)
    if len(picks) < n:
        notes.append(f"{cat}: strict pool exhausted at {len(picks)}/{n}")
    return picks, notes


def _rationale(cat: str, w: dict, l: dict, wb: dict, lb: dict) -> list[str]:
    cfg = CATEGORIES[cat]
    r = [f"{cfg['winner']} succeeds: stop {w.get('stop_dist', float('nan')):.1f} m "
         f"in {w['steps']} steps ({w['split']}, {w['map']})",
         f"{cfg['loser']} fails: {l['failure']} after {l['steps']} steps"]
    if wb["final_est_m"] is not None:
        r.append(f"{cfg['winner']} final belief estimate error "
                 f"{wb['final_est_m']:.1f} m (to target "
                 f"{wb['est_to_target_m']:.1f} m)"
                 f"{' -- converged' if wb['story_ok'] else ''}")
    if lb["max_est_m"] is not None:
        r.append(f"{cfg['loser']} max belief estimate error "
                 f"{lb['max_est_m']:.1f} m"
                 f"{' -- drifted' if lb['drift_ok'] else ''}")
    return r


# -------------------------------------------------------------------- select


def select(categories: list[str], per_category: int, count_c: int,
           overrides: list[str], excludes: list[str],
           episode_roots: dict[str, list[Path]] | None,
           alignment_paths: dict[str, Path],
           out: Path, seed: int = 0) -> dict:
    result: dict = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "seed": seed, "excluded": sorted(excludes),
                    "categories": {}}
    all_picks: list[dict] = []
    for cat in categories:
        n = count_c if cat == "C" else per_category
        align_path = alignment_paths[cat]
        if not align_path.exists():
            result["categories"][cat] = {
                "error": f"alignment csv missing: {align_path}", "pool": 0,
                "requested": n, "selected": []}
            continue
        alignment = load_alignment_csv(align_path)
        pool = [e for e in pool_for(cat, alignment, episode_roots)
                if e not in excludes]
        override_picks: list[dict] = []
        notes: list[str] = []
        for eid in overrides:
            if eid not in pool:
                notes.append(f"{eid}: override not in {cat} pool -- skipped")
                continue
            w, l, err = candidate_pair(cat, eid, episode_roots)
            if err:
                notes.append(f"{eid}: override invalid ({err}) -- skipped")
                continue
            wb, lb = belief_stats(w), belief_stats(l)
            override_picks.append({
                "category": cat, "episode_id": eid, "split": l["split"],
                "map": l["map"],
                "winner_model": CATEGORIES[cat]["winner"],
                "loser_model": CATEGORIES[cat]["loser"],
                "winner_ref": w["_path"], "loser_ref": l["_path"],
                "winner_steps": w["steps"], "loser_steps": l["steps"],
                "loser_failure": l["failure"],
                "winner_final_est_m": wb["final_est_m"],
                "winner_est_to_target_m": wb["est_to_target_m"],
                "loser_max_est_m": lb["max_est_m"],
                "rationale": _rationale(cat, w, l, wb, lb) + ["override"]})
        remaining = max(0, n - len(override_picks))
        picks, gnotes = greedy_select(
            cat, [e for e in pool if e not in overrides], remaining,
            episode_roots, seed=seed)
        selected = override_picks + picks
        result["categories"][cat] = {"pool": len(pool), "requested": n,
                                     "selected": selected,
                                     "notes": notes + gnotes}
        all_picks.extend(selected)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", default="A,B,C")
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--count-c", type=int, default=2)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[],
                        help="episode id to remove from every pool "
                             "(repeatable; recorded in selection.json)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(OUT / "selection.json"))
    args = parser.parse_args()
    cats = [c for c in args.categories.split(",") if c]
    alignment_paths = {"A": ALIGN_V4V5, "C": ALIGN_V4V5, "B": ALIGN_V4V3}
    result = select(cats, args.per_category, args.count_c, args.override,
                    args.exclude, None, alignment_paths, Path(args.out),
                    args.seed)
    for cat in cats:
        r = result["categories"][cat]
        print(f"[{cat}] pool={r['pool']} requested={r['requested']} "
              f"selected={len(r['selected'])}")
        for p in r["selected"]:
            print(f"  {p['episode_id']}  {p['winner_model']} win / "
                  f"{p['loser_model']} {p['loser_failure']} "
                  f"| winner_est {p['winner_final_est_m']} m "
                  f"| loser_max_est {p['loser_max_est_m']} m")
        for n in r.get("notes", []):
            print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
