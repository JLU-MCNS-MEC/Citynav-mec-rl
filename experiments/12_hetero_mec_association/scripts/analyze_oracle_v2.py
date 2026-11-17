#!/usr/bin/env python3
"""Aggregate and compare Oracle v1 (E2E tie-break) and v2 (agreement only).

Reads the preserved v1 results under results/baselines/<group>/oracle/seed42/
and the new v2 results under results/oracle_action_agreement_v2/<group>/.
Writes CSV tables, figures and ORACLE_V2_REPORT.md.

Safe to run while v2 is still in progress (missing groups are skipped).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GROUP_ORDER = ("two_homo", "two_hetero", "five_homo", "five_hetero")
OUT = P.RES / "oracle_action_agreement_v2"
SUM = OUT / "summary"
FIG = OUT / "figures"
MODELS = P.MODEL_ORDER
LABEL = P.MODEL_LABEL
NODES = [f"G{i}" for i in range(1, 11)]


def load(p):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def v1_journal(group):
    d = P.RES / "baselines" / group / "oracle" / f"seed{P.EVAL_SEED}"
    rows = []
    for p in sorted(d.glob("journal.part*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    ok = [r for r in rows if r.get("execution_status") == "ok"
          and r.get("steps", 0) > 0]
    return ok


def pct(values):
    a = np.asarray(values, np.float64)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99))}


def v1_row(group):
    d = P.RES / "baselines" / group / "oracle" / f"seed{P.EVAL_SEED}"
    s = load(d / "summary.json")
    if not s:
        return None
    ok = v1_journal(group)
    inf = pct([r["inf_mean_ms"] for r in ok]) if ok else {}
    comm = pct([r["comm_mean_ms"] for r in ok]) if ok else {}
    ne = pct([r["final_dist"] for r in ok]) if ok else {}
    row = {
        "group": group, "version": "oracle_e2e_tiebreak_v1",
        "n_ok": s.get("n_ok"), "n_exception": s.get("n_exception"),
        "sr": s["sr"], "osr": s["osr"], "spl_mean": s.get("spl_mean"),
        "ne_mean_m": ne.get("mean", s.get("ne_mean_m")),
        "ne_p50_m": ne.get("p50"), "ne_p95_m": ne.get("p95"),
        "e2e_mean_ms": s["e2e_mean_ms"], "e2e_p50_ms": s.get("e2e_p50_ms"),
        "e2e_p95_ms": s.get("e2e_p95_ms"), "e2e_p99_ms": s.get("e2e_p99_ms"),
        "inf_mean_ms": inf.get("mean", s.get("inf_mean_ms")),
        "inf_p50_ms": inf.get("p50"), "inf_p95_ms": inf.get("p95"),
        "comm_mean_ms": comm.get("mean", s.get("comm_mean_ms")),
        "comm_p50_ms": comm.get("p50"), "comm_p95_ms": comm.get("p95"),
        "encode_mean_ms": s.get("encode_mean_ms"),
        "exact_agreement_rate": s.get("oracle_agree_rate"),
        "at_least_one_agreement_rate": s.get("oracle_agree_rate"),
        "fallback_rate": (None if s.get("oracle_agree_rate") is None
                          else 1.0 - s["oracle_agree_rate"]),
        "tie_break_rate": None, "mean_exact_agreement_candidates": None,
        "mean_available_candidates": None,
        "oracle_agree_rate": s.get("oracle_agree_rate"),
        "failure_counts": s.get("failure_counts", {}),
    }
    for mt in MODELS:
        row[f"{LABEL[mt]}_share"] = s.get("model_step_share", {}).get(mt, 0.0)
    for g in NODES:
        row[f"{g}_share"] = s.get("node_step_share", {}).get(g, 0.0)
    return row


def v2_row(group):
    d = OUT / group
    s = load(d / "summary.json")
    if not s:
        return None
    row = {
        "group": group, "version": "oracle_action_agreement_v2",
        "n_ok": s.get("n_ok"), "n_exception": s.get("n_exception"),
        "sr": s["sr"], "osr": s["osr"], "spl_mean": s.get("spl_mean"),
        "ne_mean_m": s.get("ne_mean_m"), "ne_p50_m": s.get("ne_p50_m"),
        "ne_p95_m": s.get("ne_p95_m"),
        "e2e_mean_ms": s["e2e_mean_ms"], "e2e_p50_ms": s.get("e2e_p50_ms"),
        "e2e_p95_ms": s.get("e2e_p95_ms"), "e2e_p99_ms": s.get("e2e_p99_ms"),
        "inf_mean_ms": s.get("inf_mean_ms"),
        "inf_p50_ms": s.get("inf_p50_ms"),
        "inf_p95_ms": s.get("inf_p95_ms"),
        "comm_mean_ms": s.get("comm_mean_ms"),
        "comm_p50_ms": s.get("comm_p50_ms"),
        "comm_p95_ms": s.get("comm_p95_ms"),
        "encode_mean_ms": s.get("encode_mean_ms"),
        "exact_agreement_rate": s.get("exact_agreement_rate"),
        "at_least_one_agreement_rate": s.get("exact_agreement_rate"),
        "fallback_rate": s.get("fallback_rate"),
        "tie_break_rate": s.get("tie_break_rate"),
        "mean_exact_agreement_candidates":
            s.get("mean_exact_agreement_candidates"),
        "mean_available_candidates": s.get("mean_available_candidates"),
        "oracle_agree_rate": s.get("oracle_agree_rate"),
        "failure_counts": s.get("failure_counts", {}),
    }
    for mt in MODELS:
        row[f"{LABEL[mt]}_share"] = s.get("model_step_share", {}).get(mt, 0.0)
    for g in NODES:
        row[f"{g}_share"] = s.get("node_step_share", {}).get(g, 0.0)
    return row


COLUMNS = [
    "group", "version", "n_ok", "n_exception", "sr", "osr", "spl_mean",
    "ne_mean_m", "ne_p50_m", "ne_p95_m",
    "e2e_mean_ms", "e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms",
    "inf_mean_ms", "inf_p50_ms", "inf_p95_ms",
    "comm_mean_ms", "comm_p50_ms", "comm_p95_ms", "encode_mean_ms",
    "exact_agreement_rate", "at_least_one_agreement_rate", "fallback_rate",
    "tie_break_rate", "mean_exact_agreement_candidates",
    "mean_available_candidates", "oracle_agree_rate",
] + [f"{LABEL[m]}_share" for m in MODELS] + [f"{g}_share" for g in NODES]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fig_sr_e2e(rows):
    groups = [g for g in GROUP_ORDER
              if any(r["group"] == g for r in rows)]
    v1 = {r["group"]: r for r in rows if r["version"].endswith("v1")}
    v2 = {r["group"]: r for r in rows if r["version"].endswith("v2")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(groups))
    w = 0.36
    sr1 = [100 * v1[g]["sr"] if g in v1 else 0 for g in groups]
    sr2 = [100 * v2[g]["sr"] if g in v2 else 0 for g in groups]
    e1 = [v1[g]["e2e_mean_ms"] if g in v1 else 0 for g in groups]
    e2 = [v2[g]["e2e_mean_ms"] if g in v2 else 0 for g in groups]
    axes[0].bar(x - w / 2, sr1, w, label="v1 E2E tie-break")
    axes[0].bar(x + w / 2, sr2, w, label="v2 agreement only")
    axes[0].set_ylabel("SR (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups, rotation=20)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x - w / 2, e1, w, label="v1")
    axes[1].bar(x + w / 2, e2, w, label="v2")
    axes[1].set_ylabel("E2E mean (ms)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups, rotation=20)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Oracle v1 vs v2 (Val-Seen 2,498)")
    fig.tight_layout()
    fig.savefig(FIG / "oracle_sr_e2e.png", dpi=160)
    plt.close(fig)


def fig_inf_comm(rows):
    groups = [g for g in GROUP_ORDER
              if any(r["group"] == g for r in rows)]
    v1 = {r["group"]: r for r in rows if r["version"].endswith("v1")}
    v2 = {r["group"]: r for r in rows if r["version"].endswith("v2")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(groups))
    w = 0.36
    for ax, key, label in ((axes[0], "inf_mean_ms", "MEC inference (ms)"),
                           (axes[1], "comm_mean_ms", "Communication (ms)")):
        a = [v1[g].get(key, 0) if g in v1 else 0 for g in groups]
        b = [v2[g].get(key, 0) if g in v2 else 0 for g in groups]
        ax.bar(x - w / 2, a, w, label="v1")
        ax.bar(x + w / 2, b, w, label="v2")
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=20)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Oracle latency decomposition")
    fig.tight_layout()
    fig.savefig(FIG / "oracle_inf_comm.png", dpi=160)
    plt.close(fig)


def fig_expert_share(rows):
    groups = [g for g in GROUP_ORDER
              if any(r["group"] == g for r in rows)]
    v1 = {r["group"]: r for r in rows if r["version"].endswith("v1")}
    v2 = {r["group"]: r for r in rows if r["version"].endswith("v2")}
    labels = [LABEL[m] for m in MODELS]
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3.6),
                             sharey=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, g in zip(axes, groups):
        x = np.arange(len(MODELS))
        w = 0.36
        a = [100 * v1[g][f"{LABEL[m]}_share"] if g in v1 else 0
             for m in MODELS]
        b = [100 * v2[g][f"{LABEL[m]}_share"] if g in v2 else 0
             for m in MODELS]
        ax.bar(x - w / 2, a, w, label="v1")
        ax.bar(x + w / 2, b, w, label="v2")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(g)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("step share (%)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "oracle_expert_share.png", dpi=160)
    plt.close(fig)


def fig_agreement(rows):
    groups = [g for g in GROUP_ORDER
              if any(r["group"] == g for r in rows)]
    v1 = {r["group"]: r for r in rows if r["version"].endswith("v1")}
    v2 = {r["group"]: r for r in rows if r["version"].endswith("v2")}
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(groups))
    w = 0.2
    a = [100 * (v1[g].get("exact_agreement_rate") or 0) if g in v1 else 0
         for g in groups]
    b = [100 * (v2[g].get("exact_agreement_rate") or 0) if g in v2 else 0
         for g in groups]
    c = [100 * (v2[g].get("fallback_rate") or 0) if g in v2 else 0
         for g in groups]
    ax.bar(x - w, a, w, label="v1 at-least-one agreement")
    ax.bar(x, b, w, label="v2 exact agreement")
    ax.bar(x + w, c, w, label="v2 fallback (no exact)")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20)
    ax.set_ylabel("step share (%)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "oracle_agreement.png", dpi=160)
    plt.close(fig)


def main() -> int:
    SUM.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    v1 = [r for g in GROUP_ORDER if (r := v1_row(g))]
    v2 = [r for g in GROUP_ORDER if (r := v2_row(g))]
    rows = v1 + v2
    write_csv(SUM / "oracle_v2_summary.csv", v2)
    write_csv(SUM / "oracle_v1_vs_v2.csv", rows)
    write_csv(SUM / "oracle_v2_failures.csv", [
        {"group": r["group"], **{k: v for k, v in
                                 (r.get("failure_counts") or {}).items()}}
        for r in v2])
    if v2:
        fig_sr_e2e(rows)
        fig_inf_comm(rows)
        fig_expert_share(rows)
        fig_agreement(rows)
    print(f"v1 rows={len(v1)} v2 rows={len(v2)} -> {SUM}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
