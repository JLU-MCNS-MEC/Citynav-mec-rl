#!/usr/bin/env python3
"""09A: frozen-encoder goal matching over the sampled Stage-06 states.

For every sampled state the heading observation is rotated to north-up and
matched against the goal image with three matchers (RGB NCC, layer3 dense,
layer2 dense).  True target coordinates are used only to build the goal image
and to score the offline error.

Outputs (results/):
  09a_match_results.csv.gz   per-state, all three matchers
  09a_matching_metrics.csv   aggregate metrics + Wilson CIs
  09a_negative_controls.csv  wrong-goal controls
  09a_calibration.json       Train-Seen thresholds (frozen afterwards)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "04_visual_goal_global_map_pipeline/scripts"))
sys.path.insert(0, str(ROOT / "05_transformer_m_training/scripts"))

import geom  # noqa: E402
import matching  # noqa: E402
from dataset_v2 import RasterStore  # noqa: E402

RES = HERE.parent / "results"
BINS = ["0-20", "20-40", "40-60", "60-80", ">80"]
METHODS = ("rgb_ncc", "layer3", "layer2")
BLOCKS = {"layer3": 2, "layer2": 2}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class GoalCache:
    def __init__(self, encoder, rasters: RasterStore):
        self.encoder = encoder
        self.rasters = rasters
        self.images: dict[tuple, np.ndarray] = {}
        self.feats: dict[tuple, dict] = {}

    def key(self, name, tx, ty):
        return name, round(float(tx), 2), round(float(ty), 2)

    def image(self, name, tx, ty):
        k = self.key(name, tx, ty)
        if k not in self.images:
            self.images[k] = matching.prepare_goal(self.rasters.get(name), tx, ty)
        return k, self.images[k]

    def ensure_features(self, keys):
        todo = [k for k in keys if k not in self.feats]
        if not todo:
            return
        batch = np.stack([self.images[k] for k in todo])
        out = self.encoder.features(batch, ("layer2", "layer3"))
        for i, k in enumerate(todo):
            self.feats[k] = {"layer2": out["layer2"][i], "layer3": out["layer3"][i]}


def match_all(obs_img, obs_feat, goal_img, goal_feat):
    res = {}
    r = matching.match_rgb(obs_img, goal_img)
    res["rgb_ncc"] = r if r else (np.nan, np.nan, 0.0, 0.0)
    for layer in ("layer3", "layer2"):
        m = matching.match_dense(obs_feat[layer], goal_feat[layer], BLOCKS[layer])
        res[layer] = m
    return res


def run_chunk(rows, encoder, rasters, goals: GoalCache):
    obs = [matching.prepare_observation(rasters.get(r["map"]), r["x"], r["y"], r["yaw"])
           for r in rows]
    obs_feat = encoder.features(np.stack(obs), ("layer2", "layer3"))
    keys = [goals.image(r["map"], r["target_x"], r["target_y"])[0] for r in rows]
    goals.ensure_features(keys)
    out = []
    for i, r in enumerate(rows):
        k = keys[i]
        m = match_all(obs[i], {lay: obs_feat[lay][i] for lay in ("layer2", "layer3")},
                      goals.images[k], goals.feats[k])
        true_col, true_row = matching.offset_to_pixel(r["target_x"] - r["x"],
                                                      r["target_y"] - r["y"])
        rec = {"state_id": f"{r['episode_id']}:{r['step']}", "episode_id": r["episode_id"],
               "split": r["split"], "map": r["map"], "step": int(r["step"]),
               "distance_bin": r["distance_bin"], "true_dist_m": r["true_dist_m"],
               "in_view": int(r["target_in_view"]), "valid_fraction": r["valid_fraction"],
               "true_col": true_col, "true_row": true_row}
        for meth in METHODS:
            col, row, score, margin = m[meth]
            rec[f"{meth}_col"], rec[f"{meth}_row"] = col, row
            rec[f"{meth}_score"], rec[f"{meth}_margin"] = score, margin
            rec[f"{meth}_err_m"] = (matching.localisation_error(col, row, true_col, true_row)
                                    if np.isfinite(col) else np.nan)
        out.append(rec)
    return out


def summarise(results: list[dict], tau: dict, margin_tau: dict) -> list[dict]:
    rows = []
    for meth in METHODS:
        err = np.array([r[f"{meth}_err_m"] for r in results], float)
        score = np.array([r[f"{meth}_score"] for r in results], float)
        marg = np.array([r[f"{meth}_margin"] for r in results], float)
        inview = np.array([r["in_view"] for r in results], bool)
        good = np.isfinite(err) & (err <= 10.0)
        det = inview & good & (score >= tau[meth]) & (marg >= margin_tau[meth])
        outview = ~inview
        fp = outview & (score >= tau[meth]) & (marg >= margin_tau[meth])

        def add(scope, key, mask):
            n = int(mask.sum())
            if n == 0:
                return
            e = err[mask & inview & np.isfinite(err)]
            d = det[mask]
            rows.append({
                "method": meth, "scope": scope, "key": key, "n": n,
                "n_in_view": int((mask & inview).sum()),
                "precision_at_10": float(d.sum() / max((mask & inview & (score >= tau[meth])
                                                        & (marg >= margin_tau[meth])).sum(), 1)),
                "recall_at_10": float(d.sum() / max((mask & inview & good).sum(), 1)),
                "f1_at_10": float(2 * d.sum() / max((mask & inview & good).sum()
                                                    + (mask & inview & (score >= tau[meth])
                                                       & (marg >= margin_tau[meth])).sum(), 1)),
                "mae_m": float(np.mean(e)) if len(e) else float("nan"),
                "median_m": float(np.median(e)) if len(e) else float("nan"),
                "p90_m": float(np.percentile(e, 90)) if len(e) else float("nan"),
                "prec_5m": float(np.mean(e <= 5)) if len(e) else float("nan"),
                "prec_10m": float(np.mean(e <= 10)) if len(e) else float("nan"),
                "prec_20m": float(np.mean(e <= 20)) if len(e) else float("nan"),
                "fp_rate_out_of_view": float(fp[mask].sum() / max((mask & outview).sum(), 1)),
                "auroc": auroc(score[mask & inview], (err[mask & inview] <= 10)
                               if (mask & inview).any() else np.array([], bool)),
            })

        add("all", "all", np.ones(len(results), bool))
        for split in ("val_seen", "val_unseen"):
            add("split", split, np.array([r["split"] == split for r in results]))
        for b in BINS:
            add("distance", b, np.array([r["distance_bin"] == b for r in results]))
        for m in sorted({r["map"] for r in results}):
            add("map", m, np.array([r["map"] == m for r in results]))
        valid = np.array([r["valid_fraction"] for r in results])
        add("validity", "<0.5", valid < 0.5)
        add("validity", ">=0.5", valid >= 0.5)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(RES / "09a_match_manifest.csv"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--calibration", default=str(RES / "09a_calibration.json"))
    args = parser.parse_args()
    calib_path = Path(args.calibration)
    if not calib_path.exists():
        raise SystemExit("run calibrate_09a.py first (thresholds must come from Train-Seen)")
    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    tau = calib["tau"]
    margin_tau = calib["margin_tau"]

    with open(args.manifest, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("x", "y", "yaw", "target_x", "target_y", "true_dist_m", "valid_fraction"):
            r[k] = float(r[k])
        r["step"] = int(r["step"])
        r["target_in_view"] = int(r["target_in_view"])
    if args.limit:
        rows = rows[:args.limit]
    print(f"states: {len(rows)} | tau {tau} | margin {margin_tau}", flush=True)

    encoder = matching.Encoder(args.device)
    rasters = RasterStore()
    goals = GoalCache(encoder, rasters)
    results = []
    started = time.time()
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        results.extend(run_chunk(chunk, encoder, rasters, goals))
        if (i // args.batch) % 20 == 0:
            done = len(results)
            rate = done / max(time.time() - started, 1e-9)
            print(f"{done}/{len(rows)} states | {rate:.0f} st/s | "
                  f"elapsed {time.time() - started:.0f}s | "
                  f"eta {(len(rows) - done) / rate:.0f}s", flush=True)
    elapsed = time.time() - started

    out_gz = RES / "09a_match_results.csv.gz"
    fields = list(results[0].keys())
    with gzip.open(out_gz, "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    summary = summarise(results, tau, margin_tau)
    with (RES / "09a_matching_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    speed = {"states": len(results), "seconds": round(elapsed, 1),
             "ms_per_state": round(1000 * elapsed / len(results), 2)}
    (RES / "09a_run_meta.json").write_text(json.dumps(speed, indent=2) + "\n", encoding="utf-8")
    core = {(r["method"], r["scope"], r["key"]): r for r in summary
            if r["scope"] == "all"}
    print(json.dumps({f"{m}": {k: round(v, 4) if isinstance(v, float) else v
                               for k, v in core[(m, "all", "all")].items()
                               if k in ("n", "n_in_view", "precision_at_10",
                                        "recall_at_10", "f1_at_10", "mae_m", "median_m",
                                        "p90_m", "prec_5m", "prec_10m", "prec_20m",
                                        "fp_rate_out_of_view", "auroc")}
                     for m in METHODS}, indent=1))
    print(json.dumps(speed), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
