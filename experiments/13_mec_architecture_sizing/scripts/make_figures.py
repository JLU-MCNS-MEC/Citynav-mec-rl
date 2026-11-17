#!/usr/bin/env python3
"""Stage 13 figures: FLOPs target vs current vs feasible candidates, params, breakdown."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RES = EXP / "results"
FIG = EXP / "figures"

MODELS = ["Small", "Medium", "Transformer-S", "Transformer-M", "Large"]


def load_csv(name):
    with (RES / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    base = json.loads((RES / "baseline_flops.json").read_text())
    cand = load_csv("candidate_architectures.csv")
    tr = load_csv("weight_transfer_audit.csv")

    def pick(model, plan):
        for r in cand:
            if r["model"] == model and r["plan"] == plan:
                return r
        return None

    def pick_tr(model, plan):
        for r in tr:
            if r["model"] == model and r["plan"] == plan:
                return r
        return None

    # FLOPs target vs current vs B vs C
    x = np.arange(len(MODELS))
    w = 0.2
    cur = [base[m]["flops"] / 1e9 for m in MODELS]
    b = [float(pick(m, "B_two_param_cap")["predicted_flops"]) / 1e9
         if pick(m, "B_two_param_cap") else 0 for m in MODELS]
    c = [float(pick(m, "C_target_multi")["predicted_flops"]) / 1e9
         if pick(m, "C_target_multi") else 0 for m in MODELS]
    tgt = [base[m]["target_gflops"] for m in MODELS]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - 1.5 * w, cur, w, label="current")
    ax.bar(x - 0.5 * w, b, w, label="B: best 2-param (caps)")
    ax.bar(x + 0.5 * w, c, w, label="C: target-hitting (hidden)")
    ax.bar(x + 1.5 * w, tgt, w, label="target")
    ax.set_yscale("log")
    ax.set_ylabel("GFLOPs (log)")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.set_title("Stage-11 MEC experts: current / candidate / target FLOPs")
    fig.tight_layout()
    fig.savefig(FIG / "flops_target_vs_current.png", dpi=160)
    plt.close(fig)

    # Parameter comparison
    cur_p = [base[m]["params"] / 1e6 for m in MODELS]
    b_p = []
    c_p = []
    for m in MODELS:
        rb = pick_tr(m, "B_two_param_cap")
        rc = pick_tr(m, "C_target_multi")
        b_p.append(float(rb["params_candidate"]) / 1e6 if rb else 0)
        c_p.append(float(rc["params_candidate"]) / 1e6 if rc else 0)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - w, cur_p, w, label="current")
    ax.bar(x, b_p, w, label="B: best 2-param (caps)")
    ax.bar(x + w, c_p, w, label="C: target-hitting (hidden)")
    ax.set_yscale("log")
    ax.set_ylabel("parameters (M, log)")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.set_title("Parameter count: current vs candidate architectures")
    fig.tight_layout()
    fig.savefig(FIG / "params_comparison.png", dpi=160)
    plt.close(fig)

    # Module breakdown (grouped categories)
    rows = load_csv("module_flops_breakdown.csv")
    cats = {
        "visual": ("vis_proj", "visual_proj", "cur_proj"),
        "encoder": ("encoder", "fusion_encoder"),
        "cross_attn": ("goal_current_attn",),
        "history": ("hist_", "history_"),
        "head": ("head",),
        "other": (),
    }
    grouped = {m: defaultdict(float) for m in MODELS}
    for r in rows:
        name = r["module"]
        val = float(r["flops"])
        put = "other"
        for cat, keys in cats.items():
            if cat == "other":
                continue
            if any(k in name for k in keys):
                put = cat
                break
        grouped[r["model"]][put] += val
    cat_names = ["visual", "encoder", "cross_attn", "history", "head", "other"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bottom = np.zeros(len(MODELS))
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(cat_names)))
    for cat, col in zip(cat_names, colors):
        vals = np.asarray([grouped[m].get(cat, 0.0) / 1e6 for m in MODELS])
        ax.bar(x, vals, bottom=bottom, label=cat, color=col)
        bottom += vals
    ax.set_ylabel("MFLOPs")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Module-level FLOPs breakdown (original models)")
    fig.tight_layout()
    fig.savefig(FIG / "module_flops_breakdown.png", dpi=160)
    plt.close(fig)

    print(f"wrote {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
