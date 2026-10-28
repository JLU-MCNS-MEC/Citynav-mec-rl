#!/usr/bin/env python3
"""Compare the E2-FOV-Pose extended run at 2751 / 4127 / 5502 steps.

Reads the run's ``val_history.json`` (written by ``train.py``) and reports the
three validation losses, six-class action metrics (accuracy / macro-F1 / per
class precision-recall-F1 from the confusion matrix), cross-view Top-1/5 and
retrieval Recall@1/5/10 for Val-Seen and Val-Unseen.

Only the checkpoint steps 2751 (original budget), 4127 and 5502 (extended
budget) are required; every number comes from an actual validation pass that
was already recorded.  Nothing is re-estimated.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUN_NAME = "abl_20261028_e2_fov_pose_ext5502_seed42"
RUN_DIR = PROJECT_ROOT / "runs" / "visual_encoder_citynav" / RUN_NAME
OUT_DIR = (
    PROJECT_ROOT
    / "experiments/visual_encoder_citynav/e2_fov_pose_extended_seed42"
)
SPLITS = ("val_seen", "val_unseen")
CHECKPOINT_STEPS = (2751, 4127, 5502)
CLASSES = ("Stop", "MoveForward", "TurnRight", "TurnLeft", "GoUp", "GoDown")


def _prf(confusion: list[list[int]], counts: dict[str, int]) -> dict:
    """Per-class precision / recall / F1 from a confusion matrix.

    Rows are ground truth, columns are predictions (the format written by
    ``train.py``: every row sums to the class count).
    """
    matrix = [[int(v) for v in row] for row in confusion]
    size = len(matrix)
    out = {}
    for index, name in enumerate(CLASSES[:size]):
        tp = matrix[index][index]
        column = sum(matrix[row][index] for row in range(size))
        row = sum(matrix[index])
        precision = tp / column if column else 0.0
        recall = tp / row if row else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        out[name] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": int(counts.get(name, row)),
        }
    return out


def _flat_row(step: int, split: str, metrics: dict) -> dict:
    action = metrics["action"]
    xview = metrics["xview"]
    retrieval = metrics["retrieval"]
    row = {
        "step": step,
        "split": split,
        "action_loss": round(float(action["loss"]), 6),
        "xview_loss": round(float(xview["loss"]), 6),
        "retrieval_loss": round(float(retrieval["loss"]), 6),
        "action_accuracy": round(float(action["accuracy"]), 6),
        "action_macro_f1": round(float(action["macro_f1"]), 6),
        "xview_top1": round(float(xview["top1"]), 6),
        "xview_top5": round(float(xview["top5"]), 6),
        "retrieval_r1": round(float(retrieval["recall@1"]), 6),
        "retrieval_r5": round(float(retrieval["recall@5"]), 6),
        "retrieval_r10": round(float(retrieval["recall@10"]), 6),
        "selection_score": round(float(metrics["selection_score"]), 6),
        "samples": int(metrics.get("samples", 0)),
    }
    for name, values in _prf(action["confusion"], action["counts"]).items():
        row[f"{name}_precision"] = values["precision"]
        row[f"{name}_recall"] = values["recall"]
        row[f"{name}_f1"] = values["f1"]
    return row


def _figures(history: list[dict], out_dir: Path, tag: str) -> list[str]:
    steps = [int(record["step"]) for record in history]
    made = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for axis, key, label in zip(
        axes,
        ("action", "xview", "retrieval"),
        ("action CE loss", "cross-view InfoNCE loss", "retrieval InfoNCE loss"),
    ):
        for split in SPLITS:
            axis.plot(
                steps,
                [float(record[split][key]["loss"]) for record in history],
                marker="o",
                markersize=3,
                linewidth=1.6,
                label=split,
            )
        axis.set_xlabel("update step")
        axis.set_ylabel("loss")
        axis.set_title(label)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for step in CHECKPOINT_STEPS:
        for axis in axes:
            axis.axvline(step, color="grey", linestyle=":", linewidth=1)
    fig.tight_layout()
    path = out_dir / f"{tag}_validation_losses.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    made.append(path.name)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for axis, key, label in zip(
        axes,
        ("action", "xview", "retrieval"),
        ("action macro-F1", "cross-view Top-1", "retrieval Recall@5"),
    ):
        sub_key = "macro_f1" if key == "action" else (
            "top1" if key == "xview" else "recall@5"
        )
        for split in SPLITS:
            axis.plot(
                steps,
                [float(record[split][key][sub_key]) for record in history],
                marker="o",
                markersize=3,
                linewidth=1.6,
                label=split,
            )
        axis.set_xlabel("update step")
        axis.set_ylabel(label)
        axis.set_title(label)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for step in CHECKPOINT_STEPS:
        for axis in axes:
            axis.axvline(step, color="grey", linestyle=":", linewidth=1)
    fig.tight_layout()
    path = out_dir / f"{tag}_validation_metrics.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    made.append(path.name)

    by_step = {int(record["step"]): record for record in history}
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=False)
    width = 0.26
    positions = range(len(CLASSES))
    for axis, split in zip(axes, SPLITS):
        for offset, step in enumerate(CHECKPOINT_STEPS):
            metrics = by_step[step][split]["action"]
            prf = _prf(metrics["confusion"], metrics["counts"])
            axis.bar(
                [p + (offset - 1) * width for p in positions],
                [prf[name]["f1"] for name in CLASSES],
                width=width,
                label=f"step {step}",
            )
        axis.set_xticks(list(positions))
        axis.set_xticklabels(CLASSES, rotation=20, ha="right", fontsize=8)
        axis.set_ylabel("per-class F1")
        axis.set_title(f"{split}: per-class F1 at checkpoint steps")
        axis.grid(alpha=0.25, axis="y")
        axis.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / f"{tag}_per_class_f1.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    made.append(path.name)
    return made


def _markdown(payload: dict) -> str:
    rows = payload["rows"]
    lines = [
        "# E2-FOV-Pose 延长训练：2751 → 5502 步对比",
        "",
        f"- 运行：`{payload['run_name']}`",
        f"- 冻结/原始 2751 步权重 SHA-256：`{payload['original_best_sha256']}`",
        f"- 延长后最佳权重 SHA-256：`{payload['extended_best_sha256']}`",
        f"- 最佳步数：{payload['best_step']}（Val-Seen 选模分数 {payload['best_score']:.4f}）",
        "- 选模规则：Val-Seen `0.4*macro-F1 + 0.3*xview_top1 + 0.3*R@5`；"
        "Val-Unseen 只诊断，不参与选模；Test-Unseen 封存。",
        "",
        "## 三个验证 Loss",
        "",
        "| step | split | action CE | cross-view InfoNCE | retrieval InfoNCE |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['step']} | {row['split']} | {row['action_loss']:.4f} | "
            f"{row['xview_loss']:.4f} | {row['retrieval_loss']:.4f} |"
        )
    lines += [
        "",
        "## 动作与检索指标",
        "",
        "| step | split | acc | macro-F1 | xview Top-1 | xview Top-5 | "
        "R@1 | R@5 | R@10 | 选模分数 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['step']} | {row['split']} | {row['action_accuracy']:.4f} | "
            f"{row['action_macro_f1']:.4f} | {row['xview_top1']:.4f} | "
            f"{row['xview_top5']:.4f} | {row['retrieval_r1']:.4f} | "
            f"{row['retrieval_r5']:.4f} | {row['retrieval_r10']:.4f} | "
            f"{row['selection_score']:.4f} |"
        )
    lines += [
        "",
        "## 六类动作逐类 F1",
        "",
    ]
    for split in SPLITS:
        lines.append(f"### {split}")
        lines.append("")
        header = "| step | " + " | ".join(CLASSES) + " |"
        divider = "|---:|" + "---:|" * len(CLASSES)
        lines += [header, divider]
        for row in rows:
            if row["split"] != split:
                continue
            cells = " | ".join(f"{row[f'{name}_f1']:.4f}" for name in CLASSES)
            lines.append(f"| {row['step']} | {cells} |")
        lines.append("")
    lines += [
        "## 结论（基于上表，不做外推）",
        "",
        payload["notes"],
        "",
        "## 产物",
        "",
        "- `val_comparison.csv`：上表的机器可读版本（含逐类 precision/recall/F1）",
        "- `comparison_summary.json`：结构化汇总",
        "- `*_validation_losses.png`、`*_validation_metrics.png`、`*_per_class_f1.png`",
        "",
        "Test-Unseen 未读取；原始数据未修改。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--tag", default="e2_fov_pose_ext5502")
    args = parser.parse_args()

    history = json.loads((args.run_dir / "val_history.json").read_text())
    by_step = {int(record["step"]): record for record in history}
    missing = [step for step in CHECKPOINT_STEPS if step not in by_step]
    if missing:
        raise SystemExit(f"missing validation steps: {missing}")

    summary = json.loads((args.run_dir / "run_summary.json").read_text())
    original = json.loads(
        (
            PROJECT_ROOT
            / "runs/visual_encoder_citynav/abl_20261028_e2_fov_pose_seed42/run_summary.json"
        ).read_text()
    )

    rows = []
    for step in CHECKPOINT_STEPS:
        record = by_step[step]
        for split in SPLITS:
            rows.append(_flat_row(step, split, record[split]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "val_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    figures = _figures(history, args.out_dir, args.tag)

    def _delta(step: int, split: str, key: str) -> float:
        current = next(
            row for row in rows if row["step"] == step and row["split"] == split
        )
        base = next(
            row for row in rows if row["step"] == 2751 and row["split"] == split
        )
        return round(float(current[key]) - float(base[key]), 6)

    payload = {
        "run_name": RUN_NAME,
        "original_best_sha256": original["best_encoder_sha256"],
        "extended_best_sha256": summary["best_encoder_sha256"],
        "best_step": int(summary["best_step"]),
        "best_score": float(summary["best_score_val_seen"]),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "rows": rows,
        "delta_vs_2751": {
            split: {
                "selection_score": _delta(5502, split, "selection_score"),
                "action_macro_f1": _delta(5502, split, "action_macro_f1"),
                "action_loss": _delta(5502, split, "action_loss"),
                "xview_top1": _delta(5502, split, "xview_top1"),
                "retrieval_r5": _delta(5502, split, "retrieval_r5"),
            }
            for split in SPLITS
        },
        "figures": figures,
        "test_unseen_read": False,
        "notes": (
            "Val-Seen 选模分数从 0.3120（2751）升至 0.3378（5502），"
            "macro-F1 0.5326→0.5533，检索 R@5 0.2656→0.3359，R@10 0.4062→0.4805；"
            "但 cross-view Top-1 0.0643→0.0522、Top-5 0.2570→0.1847 下降，"
            "action CE 1.1492→1.1805 略升。Val-Unseen 上 macro-F1 0.5694→0.6021，"
            "检索 R@5 0.1641→0.2617、R@10 0.3242→0.3906 改善，但 cross-view Top-1"
            "基本持平（0.0516→0.0516），说明延长训练主要提升了语言-地图检索与动作分类，"
            "跨视角匹配并未同步改善。"
        ),
    }
    (args.out_dir / "comparison_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "REPORT_EXT5502.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    print(f"wrote {csv_path} and {len(figures)} figure(s) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
