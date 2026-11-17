#!/usr/bin/env python3
"""Stage 13-R1 figures: token contract, revised FLOPs, candidate comparison."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REV = HERE.parent
RES = REV / "results"
FIG = REV / "figures"

MODELS = ["Small", "Medium", "Transformer-S", "Transformer-M", "Large"]


def load(name):
    with (RES / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    base = {r["model"]: r for r in load("revised_baseline_flops.csv")}
    cand = load("revised_candidate_architectures.csv")

    # 1. token contract
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(MODELS))
    before = []
    after = []
    for m in MODELS:
        b = base[m]["tokens_before"]
        a = base[m]["tokens_after"]
        before.append(float(b) if b not in ("n/a", "") else 0.0)
        after.append(float(a) if a not in ("n/a", "") else 0.0)
    w = 0.36
    ax.bar(x - w / 2, before, w, label="before")
    ax.bar(x + w / 2, after, w, label="after (R1)")
    for xi, m in zip(x, MODELS):
        if base[m]["tokens_before"] in ("n/a", ""):
            ax.text(xi, 2, "MLP/cross-attn\n(all modalities)",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15)
    ax.set_ylabel("Transformer input tokens")
    ax.set_ylim(0, 118)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Token contract: 104-token unification")
    fig.tight_layout()
    fig.savefig(FIG / "token_contract_comparison.png", dpi=160)
    plt.close(fig)

    # 2. revised FLOPs vs target
    orig = [float(base[m]["flops_before"]) / 1e9 for m in MODELS]
    rev = [float(base[m]["flops_after"]) / 1e9 for m in MODELS]
    tgt = [5, 15, 40, 100, 200]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - w, orig, w, label="original")
    ax.bar(x, rev, w, label="104-token / unified")
    ax.bar(x + w, tgt, w, label="target")
    ax.set_yscale("log")
    ax.set_ylabel("GFLOPs (log)")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.set_title("Original vs unified-input FLOPs vs target")
    fig.tight_layout()
    fig.savefig(FIG / "revised_flops_comparison.png", dpi=160)
    plt.close(fig)

    # 3. candidate B/C vs target
    def pick(m, plan):
        for r in cand:
            if r["model"] == m and r["plan"] == plan:
                return float(r["predicted_flops"]) / 1e9
        return 0.0
    b = [pick(m, "B_two_param_cap") for m in MODELS]
    c = [pick(m, "C_target_multi") for m in MODELS]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - w, b, w, label="B: best 2-param (caps)")
    ax.bar(x, c, w, label="C: target-hitting (hidden)")
    ax.bar(x + w, tgt, w, label="target")
    ax.set_yscale("log")
    ax.set_ylabel("GFLOPs (log)")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.set_title("R1 candidate architectures vs target")
    fig.tight_layout()
    fig.savefig(FIG / "candidate_architecture_comparison.png", dpi=160)
    plt.close(fig)
    print(f"wrote {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
