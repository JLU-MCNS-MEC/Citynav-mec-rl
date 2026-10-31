#!/usr/bin/env python3
"""10A offline localisation evaluation: C1 vs PQ vs CA.

Identical protocol and inputs for all three matchers:

* states:    the committed 09A manifest (58,891 states sampled from the
             Stage-06 closed-loop trajectories on Val-Seen/Val-Unseen);
* inputs:    north-up 224 px observation + north-up goal image, frozen
             ResNet-50 layer2 features (fp32) -- identical to 09C;
* C1:        09C ``subpixel_match`` (dense cosine + sub-pixel centroid),
             frozen historical gate tau 0.875 (09a_calibration.json);
* PQ / CA:   trained 10A heads, tau from results/{m}/calibration.json;
* outputs:   results/eval_states.csv.gz (per-state, all methods),
             results/eval_metrics.csv (scopes: all/split/distance/length/
             validity, Wilson CIs, AUROC, params, latency),
             results/eval_run_meta.json (timings).

Resumable: states already present in eval_states.csv.gz are skipped.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common10a  # noqa: E402
from common10a import (BIN_LABELS, MANIFEST_09A, METHODS, RES, auroc,
                       require_data_root, stage06_tasks)  # noqa: E402

from model10a import load_matcher, n_params  # noqa: E402

sys.path.insert(0, str(common10a.REPO / "experiments/09_online_visual_goal_matching/scripts"))
import matching  # noqa: E402
sys.path.insert(0, str(common10a.REPO / "experiments/09_online_visual_goal_matching"
                       "/09C_continuous_localization/scripts"))
from continuous_belief import subpixel_match  # noqa: E402


def load_rows() -> list[dict]:
    tasks = stage06_tasks()
    with MANIFEST_09A.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("x", "y", "yaw", "target_x", "target_y", "true_dist_m",
                  "valid_fraction", "target_valid_fraction"):
            r[k] = float(r[k])
        r["step"] = int(r["step"])
        r["target_in_view"] = int(r["target_in_view"])
        t = tasks.get(r["episode_id"], {})
        r["length_group"] = t.get("length_group", "unknown")
        r["cohort"] = t.get("cohort", "unknown")
        r["goal_image_invalid"] = int(t.get("goal_image_invalid", False))
    return rows


def gate_metrics(results: list[dict], method: str, tau: float) -> list[dict]:
    rows = []
    err = np.array([r[f"{method}_err_m"] for r in results], float)
    score = np.array([r[f"{method}_score"] for r in results], float)
    inview = np.array([r["target_in_view"] for r in results], bool)
    good = np.isfinite(err) & (err <= 10.0)
    det = inview & good & (score >= tau)
    outview = ~inview
    fp = outview & (score >= tau)

    def add(scope, key, mask):
        n = int(mask.sum())
        if n == 0:
            return
        e = err[mask & inview & np.isfinite(err)]
        if len(e) == 0:
            return
        acc = (mask & inview & (score >= tau)).sum()
        det_n = det[mask].sum()
        good_n = (mask & inview & good).sum()
        rows.append({
            "method": method, "scope": scope, "key": key, "n": n,
            "n_in_view": int((mask & inview).sum()),
            "precision_at_10": float(det_n / max(acc, 1)),
            "recall_at_10": float(det_n / max(good_n, 1)),
            "f1_at_10": float(2 * det_n / max(good_n + acc, 1)),
            "mae_m": float(np.mean(e)), "median_m": float(np.median(e)),
            "p90_m": float(np.percentile(e, 90)),
            "prec_5m": float(np.mean(e <= 5)), "prec_10m": float(np.mean(e <= 10)),
            "prec_20m": float(np.mean(e <= 20)),
            "fp_rate_out_of_view": float(fp[mask].sum() / max((mask & outview).sum(), 1)),
            "auroc": auroc(score[mask & inview], (err[mask & inview] <= 10)
                           if (mask & inview).any() else np.array([], bool)),
        })

    add("all", "all", np.ones(len(results), bool))
    for split in ("val_seen", "val_unseen"):
        add("split", split, np.array([r["split"] == split for r in results]))
    for b in BIN_LABELS:
        add("distance", b, np.array([r["distance_bin"] == b for r in results]))
    for lg in ("short", "medium", "long"):
        add("length", lg, np.array([r["length_group"] == lg for r in results]))
    for m in sorted({r["map"] for r in results}):
        add("map", m, np.array([r["map"] == m for r in results]))
    valid = np.array([r["valid_fraction"] for r in results])
    add("validity", "<0.5", valid < 0.5)
    add("validity", ">=0.5", valid >= 0.5)
    inv = np.array([r["target_in_view"] for r in results], bool)
    add("visibility", "in_view", inv)
    add("visibility", "out_of_view", ~inv)
    add("goal_image", "invalid", np.array([r["goal_image_invalid"] for r in results], bool))
    add("goal_image", "valid", ~np.array([r["goal_image_invalid"] for r in results], bool))
    return rows


def main() -> int:
    require_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from dataset_v2 import RasterStore  # local import

    rows = load_rows()
    if args.limit:
        rows = rows[:args.limit]
    print(f"eval states: {len(rows)}", flush=True)

    # thresholds: C1 frozen historical; PQ/CA from 10A calibration
    cal09 = json.loads((common10a.REPO / "experiments/09_online_visual_goal_matching"
                        / "results/09a_calibration.json").read_text(encoding="utf-8"))
    tau = {"c1": cal09["tau"]["layer2"]}
    for m in ("pq", "ca"):
        c = json.loads((RES / m / "calibration.json").read_text(encoding="utf-8"))
        tau[m] = c["tau"]
    print(f"tau: {tau}", flush=True)

    encoder = matching.Encoder(args.device)
    rasters = RasterStore()
    models = {m: load_matcher(m, str(RES / m / "checkpoints" / "best.pt"), args.device)
              for m in ("pq", "ca")}

    out_gz = RES / "eval_states.csv.gz"
    fields = ["state_id", "episode_id", "split", "map", "step", "distance_bin",
              "length_group", "cohort", "true_dist_m", "target_in_view",
              "valid_fraction", "goal_image_invalid", "true_col", "true_row"]
    for m in METHODS:
        fields += [f"{m}_col", f"{m}_row", f"{m}_score", f"{m}_margin",
                   f"{m}_err_m"]
    done: set[str] = set()
    if out_gz.exists():
        with gzip.open(out_gz, "rt", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done.add(r["state_id"])
    pending = [r for r in rows if r["episode_id"] + f":{r['step']}" not in done]
    print(f"done {len(done)} | pending {len(pending)}", flush=True)

    # goal handling: per-batch dedup only.  Each episode appears once in the
    # manifest, so cross-batch caching never pays off and bloats RAM/VRAM
    # (~8 GB each for 5k goals, which OOM-killed the first run).
    timings = {m: [] for m in METHODS}
    t_feat = 0.0
    started = time.time()
    n = 0
    with gzip.open(out_gz, "at", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not done:
            w.writeheader()
        for i0 in range(0, len(pending), args.batch):
            chunk = pending[i0:i0 + args.batch]
            obs = [matching.prepare_observation(rasters.get(r["map"]),
                                                r["x"], r["y"], r["yaw"])
                   for r in chunk]
            gkeys = [(r["map"], round(float(r["target_x"]), 2),
                      round(float(r["target_y"]), 2)) for r in chunk]
            uniq = sorted(set(gkeys))
            gimg = {k: matching.prepare_goal(rasters.get(k[0]), k[1], k[2])
                    for k in uniq}
            t0 = time.time()
            of = encoder.features(np.stack(obs), ("layer2",))["layer2"]
            gf_np = encoder.features(np.stack([gimg[k] for k in gkeys]),
                                     ("layer2",))["layer2"]
            t_feat += time.time() - t0
            of_t = torch.from_numpy(of).to(args.device)
            gf_t = torch.from_numpy(gf_np).to(args.device)

            res = {m: {} for m in METHODS}
            for m in ("pq", "ca"):
                t0 = time.time()
                r = models[m](of_t, gf_t)
                torch.cuda.synchronize()
                timings[m].append(time.time() - t0)
                for k in ("col", "row", "score", "margin"):
                    res[m][k] = r[k].cpu().numpy()
            t0 = time.time()
            c1_rows = []
            for b in range(len(chunk)):
                m = subpixel_match(of[b], gf_np[b], 2)
                c1_rows.append((m["col"], m["row"], m["score"], m["margin"]))
            timings["c1"].append(time.time() - t0)
            res["c1"] = {"col": np.array([x[0] for x in c1_rows]),
                         "row": np.array([x[1] for x in c1_rows]),
                         "score": np.array([x[2] for x in c1_rows]),
                         "margin": np.array([x[3] for x in c1_rows])}

            for b, r in enumerate(chunk):
                true_col, true_row = matching.offset_to_pixel(
                    r["target_x"] - r["x"], r["target_y"] - r["y"])
                rec = {"state_id": f"{r['episode_id']}:{r['step']}",
                       "episode_id": r["episode_id"], "split": r["split"],
                       "map": r["map"], "step": r["step"],
                       "distance_bin": r["distance_bin"],
                       "length_group": r["length_group"], "cohort": r["cohort"],
                       "true_dist_m": r["true_dist_m"],
                       "target_in_view": r["target_in_view"],
                       "valid_fraction": r["valid_fraction"],
                       "goal_image_invalid": r["goal_image_invalid"],
                       "true_col": true_col, "true_row": true_row}
                for m in METHODS:
                    col = float(res[m]["col"][b])
                    row = float(res[m]["row"][b])
                    rec[f"{m}_col"], rec[f"{m}_row"] = col, row
                    rec[f"{m}_score"] = float(res[m]["score"][b])
                    rec[f"{m}_margin"] = float(res[m]["margin"][b])
                    rec[f"{m}_err_m"] = matching.localisation_error(
                        col, row, true_col, true_row) if np.isfinite(col) else np.nan
                w.writerow(rec)
                n += 1
            if (i0 // args.batch) % 20 == 0:
                rate = n / max(time.time() - started, 1e-9)
                print(f"{n}/{len(pending)} | {rate:.1f} st/s | elapsed "
                      f"{time.time() - started:.0f}s | eta "
                      f"{(len(pending) - n) / rate:.0f}s", flush=True)

    # ---- read back everything and summarise --------------------------------
    with gzip.open(out_gz, "rt", encoding="utf-8") as fh:
        results = list(csv.DictReader(fh))
    for r in results:
        for k in r:
            if k.startswith(("c1_", "pq_", "ca_")) or k in ("true_col", "true_row"):
                r[k] = float(r[k])
        r["target_in_view"] = int(r["target_in_view"])
        r["goal_image_invalid"] = int(r["goal_image_invalid"])
        r["valid_fraction"] = float(r["valid_fraction"])
        r["true_dist_m"] = float(r["true_dist_m"])

    metrics = []
    for m in METHODS:
        metrics.extend(gate_metrics(results, m, tau[m]))
    with (RES / "eval_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(metrics[0]))
        w.writeheader()
        w.writerows(metrics)

    params = {"c1": 0,  # C1 is a non-parametric frozen-encoder matcher
              "pq": n_params(models["pq"]), "ca": n_params(models["ca"])}
    latency_ms = {m: round(1000 * sum(timings[m]) / max(len(pending), 1), 3)
                  for m in METHODS if timings[m]}
    meta = {"states": len(results), "tau": tau, "params": params,
            "matcher_ms_per_state": latency_ms,
            "feature_ms_per_state": round(1000 * t_feat / max(len(results), 1), 3),
            "seconds": round(time.time() - started, 1),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (RES / "eval_run_meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                           encoding="utf-8")
    core = {(r["method"], r["scope"], r["key"]): r for r in metrics
            if r["scope"] == "all"}
    print(json.dumps({m: {k: v for k, v in core[(m, "all", "all")].items()
                          if k in ("n", "n_in_view", "precision_at_10",
                                   "recall_at_10", "f1_at_10", "mae_m", "median_m",
                                   "p90_m", "prec_5m", "prec_10m", "prec_20m",
                                   "fp_rate_out_of_view", "auroc")}
                      for m in METHODS}, indent=1))
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
