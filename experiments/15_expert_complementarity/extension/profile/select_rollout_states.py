#!/usr/bin/env python3
"""Stage 15-Extension D2c: pick states to spend counterfactual rollouts on.

Stage 15 showed sensitivity => action divergence, so we keep every divergent
state (where sensitivity can occur at all) and only a random sample of
agreeing states (kept as a control / natural-distribution anchor).

Because this oversamples divergent states relative to their natural 11%
prevalence, every row records `sel_prob` = P(state selected) so downstream
metrics can be importance-reweighted back to the natural distribution.

Episodes (not states) are split train/dev so that the profile never sees the
same task at fit and evaluation time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXT = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", nargs="+", required=True)
    ap.add_argument("--n-divergent", type=int, default=800)
    ap.add_argument("--n-agree", type=int, default=200)
    ap.add_argument("--dev-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE.parent / "data/branch_train_big/selection.csv"))
    args = ap.parse_args()

    parts = [pd.read_csv(p) for p in args.screen]
    df = pd.concat(parts, ignore_index=True)
    # a task may appear in both collection batches at different steps; dedupe on
    # (episode_key, step) keeping the first occurrence
    df = df.drop_duplicates(subset=["episode_key", "step"]).reset_index(drop=True)
    n_total = len(df)

    rng = np.random.default_rng(args.seed)
    div = df[df.divergent].sample(frac=1.0, random_state=args.seed)
    agr = df[~df.divergent].sample(frac=1.0, random_state=args.seed)
    n_div_take = min(args.n_divergent, len(div))
    n_agr_take = min(args.n_agree, len(agr))
    p_div = 1.0 if n_div_take >= len(div) else n_div_take / len(div)
    p_agr = n_agr_take / len(agr) if len(agr) else 0.0
    # natural prevalence of divergent states (before selection)
    prev = float(df.divergent.mean())

    sel = pd.concat([div.head(n_div_take), agr.head(n_agr_take)], ignore_index=True)
    sel["sel_prob"] = np.where(sel.divergent, p_div, p_agr)
    # natural-probability weight for reweighting back to the candidate pool
    sel["w_natural"] = (prev / np.maximum(sel.sel_prob, 1e-12)) * sel.divergent.astype(float) \
        + ((1 - prev) / np.maximum(sel.sel_prob, 1e-12)) * (~sel.divergent).astype(float)

    eps = sorted(sel.episode_key.unique())
    perm = rng.permutation(len(eps))
    n_dev = max(1, int(round(len(eps) * args.dev_frac)))
    dev_eps = set(eps[i] for i in perm[:n_dev])
    sel["split"] = np.where(sel.episode_key.isin(dev_eps), "dev", "train")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out, index=False)
    summary = {
        "candidates": n_total, "candidates_divergent_pct": prev,
        "selected": int(len(sel)),
        "selected_divergent": int(sel.divergent.sum()),
        "selected_agree": int((~sel.divergent).sum()),
        "p_div_inclusion": p_div, "p_agr_inclusion": p_agr,
        "episodes": int(len(eps)), "dev_episodes": int(n_dev),
        "dev_states": int((sel.split == "dev").sum()),
        "dev_divergent": int(((sel.split == "dev") & sel.divergent).sum()),
        "train_states": int((sel.split == "train").sum()),
        "train_divergent": int(((sel.split == "train") & sel.divergent).sum()),
        "rollouts": int(len(sel) * 5),
        "out": str(out),
    }
    print(json.dumps(summary, indent=2))
    (out.parent / "selection_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
