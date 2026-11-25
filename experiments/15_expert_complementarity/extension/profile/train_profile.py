#!/usr/bin/env python3
"""Stage 15-Extension E: lightweight expert reliability profile.

q(s, m) = P(final success | observable state s, current expert m,
            fixed continuation policy TS)

Inputs are strictly observable at decision time: UAV pose, belief statistics,
history length/last action, candidate-emptiness flag, current expert's action
confidence (from the already-required forward), and a one-hot expert id.
No true target distance, no teacher action, no final outcome at inference.

Data: snapshots_train (100 states) x 5 experts; split by episode_id.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
REPO = EXT.parents[2]
sys.path.insert(0, str(EXT.parent / "counterfactual"))
sys.path.insert(0, str(REPO / "experiments/16_cumulative_e2e_association/p1"))
sys.path.insert(0, str(REPO / "experiments/12_hetero_mec_association/scripts"))

from snapshot_io import load_snapshot  # noqa: E402

KEYS = ["small", "medium", "ts", "tm", "large"]
SNAP_DIR = EXT / "data/snapshots_train"
BRANCH = EXT / "data/branch_train/branch_rollout_results.csv"
TRACES = EXT / "data/branch_train/branch_action_traces.json"


def snapshot_features(p: Path) -> dict:
    s = load_snapshot(p)
    meta = json.loads(p.with_suffix(".json").read_text())
    pos = np.asarray(s.pos, float)
    yaw = float(s.yaw)
    if s.belief_means is not None and len(s.belief_means):
        w = np.asarray(s.belief_weights, float)
        means = np.asarray(s.belief_means, float)
        vars_ = np.asarray(s.belief_vars, float)
        est = means[int(np.argmax(w))]
        unc = float(np.sqrt(np.mean(vars_)))
        ent = float(-(w * np.log(w + 1e-12)).sum() / max(np.log(len(w)), 1e-9))
        est_dist = float(np.linalg.norm(est - pos))
        since = float(s.step - s.belief_last_update)
    else:
        unc, ent, est_dist, since = 50.0, 0.0, 0.0, 99.0
    hist_len = int(len(s.hist_act))
    last_act = int(s.hist_act[-1]) if hist_len else -1
    return {
        "episode_id": s.episode_id, "start_step": int(s.step),
        "pos_x": pos[0], "pos_y": pos[1], "yaw": yaw,
        "belief_unc": unc, "belief_ent": ent, "belief_since": since,
        "belief_est_dist": est_dist, "hist_len": hist_len,
        "last_action": last_act, "empty_cands": bool(s.empty_cands),
    }


def load_data():
    idx = pd.read_csv(SNAP_DIR / "snapshot_index.csv")
    feats = {}
    for _, r in idx.iterrows():
        f = snapshot_features(Path(r["snapshot"]))
        feats[(f["episode_id"], f["start_step"])] = f
    br = pd.read_csv(BRANCH)
    br["success"] = br["success"].astype(str).str.lower().isin(["true", "1"])
    # first-step confidence from traces
    conf = {}
    try:
        traces = json.loads(TRACES.read_text())
        for t in traces:
            a = t["actions"][0] if t["actions"] else None
            conf[(t["episode_id"], t["start_step"], t["first_expert"])] = \
                float(a["confidence"]) if a else 0.0
    except Exception:  # noqa: BLE001
        pass
    rows = []
    for _, r in br.iterrows():
        key = (r["episode_id"], int(r["start_step"]))
        if key not in feats:
            continue
        f = feats[key]
        rows.append({**{k: f[k] for k in f if k not in ("episode_id", "start_step")},
                     "episode_id": r["episode_id"], "start_step": int(r["start_step"]),
                     "expert": r["first_expert"], "success": bool(r["success"]),
                     "confidence": conf.get((r["episode_id"], int(r["start_step"]),
                                             r["first_expert"]), 0.0),
                     "cum_e2e_ms": float(r.get("final_dist_m", 0.0))})
    return pd.DataFrame(rows)


STATE_COLS = ["pos_x", "pos_y", "yaw", "belief_unc", "belief_ent",
              "belief_since", "belief_est_dist", "hist_len", "last_action",
              "empty_cands"]


def build_xy(df, cols):
    num_cols = [c for c in cols if c != "expert"]
    X = [df[num_cols].to_numpy(np.float32)]
    oh = np.zeros((len(df), 5), np.float32)
    for i, e in enumerate(df.expert):
        oh[i, KEYS.index(e)] = 1.0
    if "expert" in cols:
        X.append(oh)
    return np.concatenate(X, axis=1).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos = p[y == 1]; neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    aucs = [(a > b) + 0.5 * (a == b) for a in pos for b in neg]
    return float(np.mean(aucs))


def pr_auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    order = np.argsort(-p)
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    rec = tp / max(y.sum(), 1)
    ap = np.sum((rec - np.r_[0, rec[:-1]]) * prec)
    return float(ap)


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(y, p, bins=10):
    y = np.asarray(y); p = np.asarray(p)
    lo = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(lo[:-1], lo[1:]):
        m = (p >= a) & (p < b)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def train_mlp(Xtr, ytr, Xdv, ydv, seed=42, epochs=400, lr=3e-3):
    torch.manual_seed(seed)
    m = MLP(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.as_tensor(Xtr); yt = torch.as_tensor(ytr, dtype=torch.float32)
    for _ in range(epochs):
        m.train()
        loss = F.binary_cross_entropy_with_logits(m(Xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = torch.sigmoid(m(torch.as_tensor(Xdv))).numpy()
    return m, p


def main():
    df = load_data()
    print(f"samples={len(df)} states={df.groupby(['episode_id','start_step']).ngroups} "
          f"episodes={df.episode_id.nunique()}")
    eps = sorted(df.episode_id.unique())
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(eps))
    n_dev = max(2, int(round(len(eps) * 0.2)))
    dev_eps = set(eps[i] for i in perm[:n_dev])
    tr = df[~df.episode_id.isin(dev_eps)].reset_index(drop=True)
    dv = df[df.episode_id.isin(dev_eps)].reset_index(drop=True)
    print(f"train_eps={len(eps)-n_dev} dev_eps={n_dev} "
          f"train_samples={len(tr)} dev_samples={len(dv)}")

    results = {}
    # B1 static prior
    prior = float(tr.success.mean())
    p_static = np.full(len(dv), prior)
    results["static_prior"] = {"auc": auc(dv.success, p_static),
                               "pr_auc": pr_auc(dv.success, p_static),
                               "brier": brier(dv.success, p_static),
                               "ece": ece(dv.success, p_static)}
    # B2 confidence only
    pc = dv.confidence.to_numpy(np.float32)
    results["confidence"] = {"auc": auc(dv.success, pc),
                             "pr_auc": pr_auc(dv.success, pc),
                             "brier": brier(dv.success, pc),
                             "ece": ece(dv.success, pc)}
    # B3 state only
    Xtr = build_xy(tr, STATE_COLS); Xdv = build_xy(dv, STATE_COLS)
    _, p3 = train_mlp(Xtr, tr.success.to_numpy(), Xdv, dv.success.to_numpy())
    results["state_only"] = {"auc": auc(dv.success, p3),
                             "pr_auc": pr_auc(dv.success, p3),
                             "brier": brier(dv.success, p3),
                             "ece": ece(dv.success, p3)}
    # B4 state + expert
    cols4 = STATE_COLS + ["expert"]
    Xtr = build_xy(tr, cols4); Xdv = build_xy(dv, cols4)
    _, p4 = train_mlp(Xtr, tr.success.to_numpy(), Xdv, dv.success.to_numpy())
    results["state_expert"] = {"auc": auc(dv.success, p4),
                               "pr_auc": pr_auc(dv.success, p4),
                               "brier": brier(dv.success, p4),
                               "ece": ece(dv.success, p4)}
    # B5 state + expert + confidence (uncertainty proxy)
    cols5 = STATE_COLS + ["expert", "confidence"]
    Xtr = build_xy(tr, cols5); Xdv = build_xy(dv, cols5)
    _, p5 = train_mlp(Xtr, tr.success.to_numpy(), Xdv, dv.success.to_numpy())
    results["state_expert_conf"] = {"auc": auc(dv.success, p5),
                                    "pr_auc": pr_auc(dv.success, p5),
                                    "brier": brier(dv.success, p5),
                                    "ece": ece(dv.success, p5)}
    # sensitive-state detection (outcome divergence across experts)
    piv = df.pivot_table(index=["episode_id", "start_step"], columns="expert",
                         values="success")
    sens = (piv.nunique(axis=1) > 1)
    sens_ids = set(sens[sens].index)
    dv_ids = set(zip(dv.episode_id, dv.start_step))
    # per-state sensitivity score = spread of predicted success across experts
    state_pred = {}
    for i, r in dv.iterrows():
        state_pred.setdefault((r.episode_id, r.start_step), []).append(p5[i])
    spread = {k: float(np.max(v) - np.min(v)) for k, v in state_pred.items()}
    y_sens = np.asarray([1 if k in sens_ids else 0 for k in spread])
    s_score = np.asarray(list(spread.values()))
    results["sensitive_detection"] = {
        "n_dev_states": len(spread), "n_sensitive": int(y_sens.sum()),
        "auc": auc(y_sens, s_score) if y_sens.sum() else None,
        "pr_auc": pr_auc(y_sens, s_score) if y_sens.sum() else None,
    }
    out = {"data": {"samples": len(df), "states": int(len(piv)),
                    "episodes": int(df.episode_id.nunique()),
                    "sensitive_states": int(sens.sum()),
                    "sensitive_pct": float(sens.mean()),
                    "train_states": int(len(tr)//5), "dev_states": int(len(dv)//5)},
           "metrics": results}
    (HERE / "profile_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
