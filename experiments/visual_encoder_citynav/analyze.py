#!/usr/bin/env python3
"""Create required CSVs, summary JSON and high-resolution PNG curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _plot_metric(rows, key, title, ylabel, path, splits=None):
    steps = [int(row["step"]) for row in rows]
    values = [_float(row.get(key)) for row in rows]
    plt.figure(figsize=(8, 4.8))
    plt.plot(steps, values, marker="o", markersize=3, linewidth=1.8)
    plt.xlabel("update step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _plot_val_history(history, path):
    steps = [int(record["step"]) for record in history]
    metrics = [
        ("action", "accuracy", "action accuracy"),
        ("action", "macro_f1", "action macro-F1"),
        ("xview", "top1", "current-view Top-1"),
        ("xview", "top5", "current-view Top-5"),
        ("retrieval", "recall@1", "retrieval Recall@1"),
        ("retrieval", "recall@5", "retrieval Recall@5"),
        ("retrieval", "recall@10", "retrieval Recall@10"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for axis, (group, key, label) in zip(axes.flat, metrics):
        for split, color in (("val_seen", "tab:blue"), ("val_unseen", "tab:orange")):
            values = []
            for record in history:
                value = record.get(split, {}).get(group, {}).get(key)
                values.append(np.nan if value is None else float(value))
            axis.plot(steps, values, marker="o", label=split, color=color)
        axis.set_title(label)
        axis.set_xlabel("step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_val_losses(history, path):
    steps = [int(record["step"]) for record in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for axis, (group, key, title) in zip(
        axes,
        [
            ("action", "loss", "val action loss"),
            ("xview", "loss", "val cross-view loss"),
            ("retrieval", "loss", "val retrieval loss"),
        ],
    ):
        for split, color in (("val_seen", "tab:blue"), ("val_unseen", "tab:orange")):
            values = [
                (
                    np.nan
                    if record.get(split, {}).get(group, {}).get(key) is None
                    else float(record[split][group][key])
                )
                for record in history
            ]
            axis.plot(steps, values, marker="o", label=split, color=color)
        axis.set_title(title)
        axis.set_xlabel("step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_per_class(history, path):
    best = max(history, key=lambda item: item["val_seen"].get("selection_score", -1))
    names = list(best["val_seen"]["action"]["per_class_accuracy"].keys())
    seen = [best["val_seen"]["action"]["per_class_accuracy"][name] for name in names]
    unseen = [best["val_unseen"]["action"]["per_class_accuracy"][name] for name in names]
    x = np.arange(len(names))
    plt.figure(figsize=(9, 4.6))
    plt.bar(x - 0.18, seen, width=0.36, label="val_seen")
    plt.bar(x + 0.18, unseen, width=0.36, label="val_unseen")
    plt.xticks(x, names, rotation=20)
    plt.ylabel("accuracy")
    plt.title(f"per-class action accuracy at best step {best['step']}")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="defaults to runs/visual_encoder_citynav/<run-name>",
    )
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "figures")
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT_ROOT / "results")
    args = parser.parse_args()
    run_dir = args.run_dir or (
        PROJECT_ROOT / "runs" / "visual_encoder_citynav" / args.run_name
    )
    output_dir = args.output_dir
    results_dir = args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = _read_csv(run_dir / "train_metrics.csv")
    history = json.loads((run_dir / "val_history.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))

    _plot_metric(
        metrics, "total_loss", "Training losses", "loss",
        output_dir / f"{args.run_name}_loss_total.png",
    )
    plt.figure(figsize=(8, 4.8))
    steps = [int(row["step"]) for row in metrics]
    for key, label in (
        ("action_loss", "action"),
        ("xview_loss", "cross-view"),
        ("retrieval_loss", "retrieval"),
    ):
        plt.plot(
            steps,
            [_float(row[key]) for row in metrics],
            marker="o", markersize=3, label=label,
        )
    plt.xlabel("update step")
    plt.ylabel("loss")
    plt.title("Independent training losses")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{args.run_name}_loss_components.png", dpi=220)
    plt.close()
    _plot_metric(
        metrics, "encoder_grad_norm", "Encoder gradient norm", "L2 norm",
        output_dir / f"{args.run_name}_encoder_grad_norm.png",
    )
    _plot_metric(
        metrics, "encoder_param_delta", "Encoder parameter change from init", "L2",
        output_dir / f"{args.run_name}_encoder_param_delta.png",
    )
    _plot_metric(
        metrics, "render_seconds_mean", "FPV render time", "seconds/sample",
        output_dir / f"{args.run_name}_render_seconds.png",
    )
    _plot_metric(
        metrics, "valid_pixel_ratio_mean", "FPV valid-pixel ratio", "ratio",
        output_dir / f"{args.run_name}_valid_pixel_ratio.png",
    )
    _plot_metric(
        metrics, "lr_encoder", "Encoder learning rate", "lr",
        output_dir / f"{args.run_name}_lr_encoder.png",
    )
    _plot_metric(
        metrics, "samples_per_second", "Training throughput", "samples/s",
        output_dir / f"{args.run_name}_throughput.png",
    )
    _plot_metric(
        metrics, "gpu_allocated_mb", "GPU allocated memory", "MiB",
        output_dir / f"{args.run_name}_gpu_memory.png",
    )
    _plot_val_history(history, output_dir / f"{args.run_name}_validation.png")
    _plot_val_losses(history, output_dir / f"{args.run_name}_validation_losses.png")
    _plot_per_class(history, output_dir / f"{args.run_name}_per_class.png")

    summary_rows = []
    for record in history:
        for split in ("val_seen", "val_unseen"):
            item = record.get(split, {})
            summary_rows.append(
                {
                    "step": record["step"],
                    "split": split,
                    "action_accuracy": item.get("action", {}).get("accuracy"),
                    "action_macro_f1": item.get("action", {}).get("macro_f1"),
                    "action_loss": item.get("action", {}).get("loss"),
                    "xview_top1": item.get("xview", {}).get("top1"),
                    "xview_top5": item.get("xview", {}).get("top5"),
                    "xview_loss": item.get("xview", {}).get("loss"),
                    "retrieval_recall@1": item.get("retrieval", {}).get("recall@1"),
                    "retrieval_recall@5": item.get("retrieval", {}).get("recall@5"),
                    "retrieval_recall@10": item.get("retrieval", {}).get("recall@10"),
                    "retrieval_loss": item.get("retrieval", {}).get("loss"),
                    "valid_pixel_ratio_mean": item.get("valid_pixel_ratio_mean"),
                    "render_seconds_mean": item.get("render_seconds_mean"),
                    "selection_score": item.get("selection_score"),
                }
            )
    summary_path = results_dir / f"{args.run_name}_validation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    loss_keys = ("step", "total_loss", "action_loss", "xview_loss", "retrieval_loss")
    loss_path = results_dir / f"{args.run_name}_loss_components.csv"
    with loss_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(loss_keys))
        writer.writeheader()
        writer.writerows({key: row[key] for key in loss_keys} for row in metrics)

    def _tail_mean(key, tail=100):
        values = [_float(row.get(key)) for row in metrics[-tail:]]
        values = [value for value in values if not np.isnan(value)]
        return float(np.mean(values)) if values else None

    resource_summary = {
        "steps": summary.get("steps"),
        "total_steps_planned": summary.get("total_steps_planned"),
        "batch_size": summary.get("config", {}).get("batch_size"),
        "train_records": summary.get("train_manifest", {}).get("records"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "gpu_max_allocated_mb": summary.get("gpu_max_allocated_mb"),
        "render_seconds_mean": _tail_mean("render_seconds_mean"),
        "valid_pixel_ratio_mean": _tail_mean("valid_pixel_ratio_mean"),
        "step_seconds_mean": _tail_mean("step_seconds"),
        "samples_per_second_mean": _tail_mean("samples_per_second"),
        "encoder_grad_norm_mean": _tail_mean("encoder_grad_norm"),
        "encoder_param_delta_final": _float(metrics[-1].get("encoder_param_delta")),
    }

    def _relative(value):
        prefix = str(PROJECT_ROOT) + "/"
        if isinstance(value, str):
            return value[len(prefix):] if value.startswith(prefix) else value
        if isinstance(value, dict):
            return {key: _relative(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_relative(item) for item in value]
        return value
    analysis = _relative({
        "run_name": args.run_name,
        "best_step": summary.get("best_step"),
        "best_score_val_seen": summary.get("best_score_val_seen"),
        "best_encoder": summary.get("best_encoder"),
        "final_encoder": summary.get("final_encoder"),
        "initialization": summary.get("initialization"),
        "action_counts": summary.get("action_counts"),
        "class_weights": summary.get("class_weights"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "gpu_max_allocated_mb": summary.get("gpu_max_allocated_mb"),
        "figures": sorted(path.name for path in output_dir.glob(f"{args.run_name}_*.png")),
        "validation_csv": str(summary_path.relative_to(PROJECT_ROOT)),
        "loss_components_csv": str(loss_path.relative_to(PROJECT_ROOT)),
        "resource_summary": resource_summary,
    })
    best = max(
        history, key=lambda item: item["val_seen"].get("selection_score", -1)
    )
    per_map_rows = []
    for split in ("val_seen", "val_unseen"):
        for map_name, item in best.get(split, {}).get("per_map", {}).items():
            per_map_rows.append(
                {
                    "step": best["step"],
                    "split": split,
                    "map_name": map_name,
                    "frames": item.get("frames"),
                    "action_accuracy": item.get("action_accuracy"),
                    "current_top1": item.get("current_top1"),
                    "current_top5": item.get("current_top5"),
                    "retrieval_recall@1": item.get("retrieval_recall@1"),
                    "retrieval_recall@5": item.get("retrieval_recall@5"),
                    "retrieval_recall@10": item.get("retrieval_recall@10"),
                    "valid_pixel_ratio_mean": item.get("valid_pixel_ratio_mean"),
                    "candidate_count": item.get("candidate_count"),
                }
            )
    per_map_path = results_dir / f"{args.run_name}_per_map.csv"
    with per_map_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_map_rows[0]))
        writer.writeheader()
        writer.writerows(per_map_rows)
    analysis["per_map_csv"] = str(per_map_path.relative_to(PROJECT_ROOT))
    analysis["per_map_rows"] = len(per_map_rows)
    (results_dir / f"{args.run_name}_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"analysis written for {args.run_name}")
    print(f"  best_step={analysis['best_step']} score={analysis['best_score_val_seen']}")
    print(f"  figures={len(analysis['figures'])} results={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
