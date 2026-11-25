#!/usr/bin/env python3
"""Stage 15-Extension E (v2): expert reliability profile on the expanded set.

q(s, m) = P(final success | observable state s, current expert m,
            fixed TS continuation), split by *episode* so no task leaks.

Compared with the 100-state smoke run this version
  * uses the divergence-screened, expanded rollout set,
  * reports the natural-distribution estimate via `w_natural`
    (the selection oversamples divergent states ~8x), and
  * evaluates sensitive-state detection, which is the property that actually
    matters for a router: does the profile know *where* expert choice matters?

All features are observable at decision time.  The consensus block uses the
five first-step expert actions, which cost five forwards (~50 ms) and are
already available to any router; no rollout outcome is used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
REPO = EXT.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXT.parent / "counterfactual"))
sys.path.insert(0, str(REPO / "experiments/16_cumulative_e2e_association/p1"))
sys.path.insert(0, str(REPO / "experiments/12_hetero_mec_association/scripts"))

from train_profile import (KEYS, MLP, auc, brier, build_xy, ece,  # noqa: E402
                           pr_auc, snapshot_features, train_mlp)
from snapshot_io import load_snapshot  # noqa: E402

STATE_COLS = ["pos_x", "pos_y", "yaw", "belief_unc", "belief_ent",
              "belief_since", "belief_est_dist", "hist_len", "last_action",
              "empty_cands"]


def resolve(p):
    """Index paths are written relative to either the repo root or the
    extension directory depending on the launch directory."""
    p = Path(p)
    if p.is_absolute() and p.exists():
        return p
    for base in (REPO, EXT, Path.cwd()):
        cand = base / p
        if cand.exists():
            return cand
    return p if p.is_absolute() else REPO / p


def load_data(screen_csv, branch_csv):
    sc = pd.read_csv(screen_csv)
    feats = {}
    for p in sc.snapshot.unique():
        f = snapshot_features(resolve(p))
        feats[(f["episode_id"], f["start_step"])] = f
    sc["key"] = [(str(e), int(s)) for e, s in zip(sc.episode_id, sc.step)]
    sc = sc.set_index("key")
    scd = sc.to_dict("index")

    br = pd.read_csv(branch_csv)
    br["success"] = br["success"].astype(str).str.lower().isin(["true", "1"])
    rows = []
    for _, r in br.iterrows():
        key = (r["episode_id"], int(r["start_step"]))
        if key not in feats:
            continue
        f = feats[key]
        srow = scd.get((str(key[0]), int(key[1])))
        if srow is None:
            continue
        e = r["first_expert"]
        d = {k: f[k] for k in f if k not in ("episode_id", "start_step")}
        d.update({
            "episode_id": r["episode_id"], "start_step": int(r["start_step"]),
            "expert": e, "success": bool(r["success"]),
            "confidence": float(srow[f"conf_{e}"]),
            "entropy": float(srow[f"ent_{e}"]),
            "n_distinct": int(srow["n_distinct_actions"]),
            "divergent": bool(srow["divergent"]),
            "sel_prob": float(srow["sel_prob"]),
            "w_natural": float(srow["w_natural"]),
        })
        for k in KEYS:                       # five-expert action vector (consensus)
            d[f"act_{k}"] = int(srow[f"a_{k}"])
        rows.append(d)
    return pd.DataFrame(rows), sc


CONSENSUS_COLS = [f"act_{k}" for k in KEYS] + ["n_distinct"]


def weighted_brier(y, p, w):
    w = np.asarray(w, float)
    return float(np.sum(w * (np.asarray(p) - np.asarray(y)) ** 2) / w.sum())


def weighted_ece(y, p, w, bins=10):
    y = np.asarray(y); p = np.asarray(p); w = np.asarray(w, float)
    lo = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(lo[:-1], lo[1:]):
        m = (p >= a) & (p < b)
        if m.sum() and w[m].sum() > 0:
            e += w[m].sum() / w.sum() * abs(
                np.average(y[m], weights=w[m]) - np.average(p[m], weights=w[m]))
    return float(e)


def weighted_auc(y, p, w):
    y = np.asarray(y); p = np.asarray(p); w = np.asarray(w, float)
    pos = w[y == 1]; neg = w[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    pp = p[y == 1]; pn = p[y == 0]
    num = 0.0
    for i, a in enumerate(pp):
        num += pos[i] * float(np.sum(neg * ((a > pn) + 0.5 * (a == pn))))
    return float(num / (pos.sum() * neg.sum()))


def metrics(y, p, w=None):
    out = {"auc": auc(y, p), "pr_auc": pr_auc(y, p),
           "brier": brier(y, p), "ece": ece(y, p)}
    if w is not None:
        out.update({"w_auc": weighted_auc(y, p, w),
                    "w_brier": weighted_brier(y, p, w),
                    "w_ece": weighted_ece(y, p, w)})
    return out


def main():
    ap = argparse.ArgumentParser()
    # NOTE: --screen must be the *selection* CSV produced by
    # select_rollout_states.py (it carries the screen columns plus split,
    # sel_prob and w_natural).
    ap.add_argument("--screen", default=str(EXT / "data/branch_train_big/selection.csv"))
    ap.add_argument("--branch", default=str(EXT / "data/branch_train_big/branch_rollout_results.csv"))
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE / "profile_results_v2.json"))
    args = ap.parse_args()

    df, sc = load_data(args.screen, args.branch)
    split = {(str(r.episode_id), int(r.step)): r.split for r in sc.reset_index().itertuples()}
    df["split"] = [split[(e, int(s))] for e, s in zip(df.episode_id, df.start_step)]
    tr = df[df.split == "train"].reset_index(drop=True)
    dv = df[df.split == "dev"].reset_index(drop=True)
    ytr = tr.success.to_numpy(); ydv = dv.success.to_numpy()
    print(f"samples={len(df)} states={df.groupby(['episode_id','start_step']).ngroups} "
          f"episodes={df.episode_id.nunique()} | train={len(tr)} dev={len(dv)} "
          f"| train SR={ytr.mean():.3f} dev SR={ydv.mean():.3f}")

    results = {}
    p_static = np.full(len(dv), float(ytr.mean()))
    results["static_prior"] = {"all": metrics(ydv, p_static),
                               "natural": metrics(ydv, p_static, dv.w_natural)}
    pc = dv.confidence.to_numpy(np.float32)
    results["confidence"] = {"all": metrics(ydv, pc),
                             "natural": metrics(ydv, pc, dv.w_natural)}

    models = {
        "state_only": STATE_COLS,
        "state_expert": STATE_COLS + ["expert"],
        "state_expert_conf": STATE_COLS + ["expert", "confidence"],
        "state_expert_conf_consensus": STATE_COLS + ["expert", "confidence",
                                                     "entropy", "divergent"] + CONSENSUS_COLS,
    }
    preds = {}
    for name, cols in models.items():
        Xtr = build_xy(tr, cols); Xdv = build_xy(dv, cols)
        _, p = train_mlp(Xtr, ytr, Xdv, ydv, seed=args.seed, epochs=args.epochs)
        preds[name] = p
        results[name] = {"all": metrics(ydv, p),
                         "natural": metrics(ydv, p, dv.w_natural)}

    # ---- sensitive-state detection -------------------------------------
    piv = df.pivot_table(index=["episode_id", "start_step"], columns="expert",
                         values="success")
    sens = (piv.nunique(axis=1) > 1)
    sens_ids = set(sens[sens].index)
    det = {}
    for name, p in preds.items():
        tmp = dv.copy(); tmp["pred"] = p
        g = tmp.groupby(["episode_id", "start_step"]).pred
        spread = (g.max() - g.min())
        y_s = np.asarray([1 if k in sens_ids else 0 for k in spread.index])
        s_s = spread.to_numpy()
        det[name] = {"states": int(len(y_s)), "positives": int(y_s.sum()),
                     "auc": auc(y_s, s_s) if y_s.sum() else None,
                     "pr_auc": pr_auc(y_s, s_s) if y_s.sum() else None}
    # consensus-based detector (no learning): number of distinct first-step actions
    st = dv.groupby(["episode_id", "start_step"]).n_distinct.max()
    y_s = np.asarray([1 if k in sens_ids else 0 for k in st.index])
    det["consensus_rule"] = {"states": int(len(y_s)), "positives": int(y_s.sum()),
                             "auc": auc(y_s, st.to_numpy()) if y_s.sum() else None,
                             "pr_auc": pr_auc(y_s, st.to_numpy()) if y_s.sum() else None}

    # ---- per-expert calibration (best model) ----------------------------
    best = "state_expert_conf_consensus"
    per_expert = {}
    for e in KEYS:
        m = dv.expert == e
        if m.sum() and ydv[m].min() != ydv[m].max():
            per_expert[e] = {"n": int(m.sum()), "sr": float(ydv[m].mean()),
                             "auc": auc(ydv[m], preds[best][m]),
                             "brier": brier(ydv[m], preds[best][m]),
                             "ece": ece(ydv[m], preds[best][m])}

    out = {
        "data": {
            "samples": int(len(df)),
            "states": int(df.groupby(["episode_id", "start_step"]).ngroups),
            "episodes": int(df.episode_id.nunique()),
            "sensitive_states": int(sens.sum()),
            "sensitive_pct": float(sens.mean()),
            "sensitive_in_train": int(sum(1 for k in sens_ids if split[k] == "train")),
            "sensitive_in_dev": int(sum(1 for k in sens_ids if split[k] == "dev")),
            "train_samples": int(len(tr)), "dev_samples": int(len(dv)),
            "train_dev_episodes": [int(tr.episode_id.nunique()), int(dv.episode_id.nunique())],
        },
        "metrics": results,
        "sensitive_detection": det,
        "per_expert_best_model": per_expert,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["data"], indent=2))
    print(json.dumps({"detection": det}, indent=2))


if __name__ == "__main__":
    main()
