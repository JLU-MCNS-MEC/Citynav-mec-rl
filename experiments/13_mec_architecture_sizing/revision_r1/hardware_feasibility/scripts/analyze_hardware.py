#!/usr/bin/env python3
"""Stage 13-R3: turn the benchmark CSV into memory / batch / latency / time tables."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
HF = HERE.parent
RES = HF / "results"
FIG = HF / "figures"
MODELS = ["Small", "Medium", "Transformer-S", "Transformer-M", "Large"]
DATA_H = {"Small": 0.26, "Medium": 0.48, "Transformer-S": 0.47,
          "Transformer-M": 0.45, "Large": 0.47}
N_STATES = 1_279_882


def load():
    with (RES / "benchmark_results.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("batch",):
            r[k] = int(r[k])
        for k in ("params", "weights_fp32_mb", "weights_bf16_mb", "forward_ms",
                  "forward_per_sample_ms", "forward_peak_mb", "fwd_bwd_ms",
                  "fwd_bwd_peak_mb", "optimizer_states_mb", "est_train_peak_mb"):
            r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else None
    return rows


def memory_table(rows):
    out = []
    for m in MODELS:
        r = next((x for x in rows if x["model"] == m and x["batch"] == 128),
                 None)
        if r is None:
            continue
        p = r["params"]
        out.append({
            "model": m, "params": int(p),
            "bf16_weights_mb": p * 2 / 1e6,
            "fp16_weights_mb": p * 2 / 1e6,
            "fp32_weights_mb": p * 4 / 1e6,
            "bf16_grads_mb": p * 2 / 1e6,
            "fp32_grads_mb": p * 4 / 1e6,
            "adamw_fp32_mb": p * 8 / 1e6,
            "fp32_master_mb": p * 4 / 1e6,
            "standard_amp_states_mb": p * 16 / 1e6,
            "bf16_no_master_states_mb": p * 12 / 1e6,
            "forward_peak_mb": r["forward_peak_mb"],
            "fwd_bwd_peak_mb": r["fwd_bwd_peak_mb"],
            "est_train_peak_mb": r["est_train_peak_mb"],
            "fits_32gb": r["fits_32gb"],
        })
    return out


def batch_table(rows):
    out = []
    for m in MODELS:
        for r in [x for x in rows if x["model"] == m]:
            out.append({
                "model": m, "batch": r["batch"],
                "fwd_peak_mb": r["forward_peak_mb"],
                "fwd_bwd_peak_mb": r["fwd_bwd_peak_mb"],
                "est_train_peak_mb": r["est_train_peak_mb"],
                "fits_32gb": r["fits_32gb"],
                "needs_gradient_checkpointing":
                    "yes" if (r["est_train_peak_mb"] is None
                              or r["est_train_peak_mb"] > 28000) else "no",
                "status": r["status"],
            })
    return out


def latency_table(rows):
    out = []
    for m in MODELS:
        for r in [x for x in rows if x["model"] == m]:
            out.append({
                "model": m, "batch": r["batch"],
                "forward_ms": r["forward_ms"],
                "forward_per_sample_ms": r["forward_per_sample_ms"],
                "fwd_bwd_ms": r["fwd_bwd_ms"],
                "fwd_bwd_per_sample_ms":
                    (r["fwd_bwd_ms"] / r["batch"] if r["fwd_bwd_ms"] else None),
                "status": r["status"],
            })
    return out


def training_table(rows):
    """Recommended batch: largest that fits with >=3GB headroom; else 64/32."""
    out = []
    for m in MODELS:
        rs = [x for x in rows if x["model"] == m]
        fitting = [x for x in rs if x["est_train_peak_mb"] is not None
                   and x["est_train_peak_mb"] < 28000]
        rec = max(fitting, key=lambda x: x["batch"]) if fitting else \
            min([x for x in rs if x["batch"] <= 64],
                key=lambda x: -x["batch"])
        b = rec["batch"]
        steps = int(np.ceil(N_STATES / b))
        fb_ms = rec["fwd_bwd_ms"]
        if fb_ms is None:
            out.append({"model": m, "batch": b, "steps_per_epoch": steps,
                        "status": "no measured step"})
            continue
        compute_h = steps * fb_ms / 1000 / 3600
        data_h = DATA_H[m]
        epoch_h = compute_h + data_h
        out.append({
            "model": m, "batch": b, "steps_per_epoch": steps,
            "fwd_bwd_ms_per_step": fb_ms,
            "compute_h_per_epoch": compute_h,
            "data_h_per_epoch": data_h,
            "epoch_h": epoch_h,
            "twenty_epoch_h": 20 * epoch_h,
            "early_stop_15ep_h": 15 * epoch_h,
            "status": "ok",
        })
    return out


def write_csv(name, rows):
    if not rows:
        return
    with (RES / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def figs(mem, batch, lat, train):
    FIG.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(mem))
    # memory
    fig, ax = plt.subplots(figsize=(10, 4.6))
    # Standard AMP regime measured by the benchmark: FP32 params + BF16 autocast.
    b = np.asarray([r["fp32_weights_mb"] for r in mem]) / 1024
    g = np.asarray([r["fp32_grads_mb"] for r in mem]) / 1024
    o = np.asarray([r["adamw_fp32_mb"] for r in mem]) / 1024
    peak = np.asarray([(r["fwd_bwd_peak_mb"] or 0) for r in mem]) / 1024
    a = np.maximum(peak - b - g, 0)
    ax.bar(x, b, label="FP32 weights (AMP)")
    ax.bar(x, g, bottom=b, label="FP32 grads")
    ax.bar(x, o, bottom=b + g, label="AdamW FP32 m/v")
    ax.bar(x, a, bottom=b + g + o, label="activations+workspace")
    ax.set_xticks(x)
    ax.set_xticklabels([r["model"] for r in mem], rotation=15)
    ax.set_ylabel("GB")
    ax.axhline(32, color="r", ls="--", lw=1, label="32 GB")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Estimated training memory at batch 128")
    fig.tight_layout()
    fig.savefig(FIG / "gpu_memory_comparison.png", dpi=160)
    plt.close(fig)

    # batch feasibility
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for m in MODELS:
        rs = [r for r in batch if r["model"] == m]
        xs = [r["batch"] for r in rs]
        ys = [(r["est_train_peak_mb"] or 0) / 1024 for r in rs]
        ax.plot(xs, ys, marker="o", label=m)
    ax.axhline(32, color="r", ls="--", lw=1)
    ax.set_xlabel("batch size")
    ax.set_ylabel("estimated training peak (GB)")
    ax.set_title("Batch-size feasibility (32 GB RTX 5090)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "batch_size_feasibility.png", dpi=160)
    plt.close(fig)

    # latency
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for m in MODELS:
        rs = [r for r in lat if r["model"] == m
              and r["forward_per_sample_ms"] is not None]
        xs = [r["batch"] for r in rs]
        ys = [r["forward_per_sample_ms"] for r in rs]
        ax.plot(xs, ys, marker="o", label=m)
    ax.set_xlabel("batch size")
    ax.set_ylabel("forward per-sample latency (ms)")
    ax.set_title("Random-input forward latency (BF16 autocast)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "forward_latency_comparison.png", dpi=160)
    plt.close(fig)

    # training time
    fig, ax = plt.subplots(figsize=(9, 4.4))
    names = [r["model"] for r in train if r.get("twenty_epoch_h")]
    tw = [r["twenty_epoch_h"] for r in train if r.get("twenty_epoch_h")]
    es = [r["early_stop_15ep_h"] for r in train if r.get("early_stop_15ep_h")]
    xx = np.arange(len(names))
    ax.bar(xx - 0.2, tw, 0.4, label="20 epochs")
    ax.bar(xx + 0.2, es, 0.4, label="early stop (15 ep)")
    ax.set_xticks(xx)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("hours")
    ax.set_title("Estimated training time")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "training_time_comparison.png", dpi=160)
    plt.close(fig)


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    rows = load()
    mem = memory_table(rows)
    batch = batch_table(rows)
    lat = latency_table(rows)
    train = training_table(rows)
    write_csv("memory_estimation.csv", mem)
    write_csv("batch_size_comparison.csv", batch)
    write_csv("forward_latency_estimation.csv", lat)
    write_csv("training_time_estimation.csv", train)
    figs(mem, batch, lat, train)
    print(f"wrote {RES} and {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
