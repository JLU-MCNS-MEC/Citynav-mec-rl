"""Aggregate E2-BEV / E2-FOV-Pose / E2-BEV-POSE against the E2 reference.

Produces:
  results/bev_pose_comparison.csv / .json
  figures/bev_pose_metrics.png
  figures/bev_pose_confusions.png
  figures/bev_pose_losses.png

Per-class precision/recall/F1 and the left/right confusion are computed from
the confusion matrix stored in val_history.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNS = {
    "E2-ref": "abl_20261028_e2_seed42",
    "E2-BEV": "abl_20261028_e2_bev_seed42",
    "E2-FOV-Pose": "abl_20261028_e2_fov_pose_seed42",
    "E2-BEV-POSE": "abl_20261028_e2_bev_pose_seed42",
}
CLASSES = ["Stop", "MoveForward", "TurnRight", "TurnLeft", "GoUp", "GoDown"]


def _load(run: str) -> tuple[dict, list, list]:
    root = PROJECT_ROOT / "runs" / "visual_encoder_citynav" / run
    history = json.loads((root / "val_history.json").read_text())
    best = max(history, key=lambda r: float(r["val_seen"].get("selection_score", -1)))
    rows = []
    metrics_csv = root / "train_metrics.csv"
    if metrics_csv.exists():
        with metrics_csv.open() as f:
            rows = list(csv.DictReader(f))
    return best, history, rows


def _prf(confusion: np.ndarray) -> dict:
    out = {}
    for i, name in enumerate(CLASSES):
        tp = float(confusion[i, i])
        fp = float(confusion[:, i].sum() - tp)
        fn = float(confusion[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(confusion[i, :].sum()),
            "predicted": int(confusion[:, i].sum()),
            "correct": int(confusion[i, i]),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    out = args.out_dir

    summary, records, curves = [], {}, {}
    for label, run in RUNS.items():
        if not (PROJECT_ROOT / "runs/visual_encoder_citynav" / run / "val_history.json").exists():
            print(f"missing {label}")
            continue
        best, history, rows = _load(run)
        vs, vu = best["val_seen"], best["val_unseen"]
        conf = np.asarray(vs["action"]["confusion"], dtype=np.int64)
        entry = {
            "run": label,
            "best_step": int(best["step"]),
            "val_seen_selection_score": float(vs.get("selection_score", -1)),
            "val_seen_action_acc": vs["action"]["accuracy"],
            "val_seen_macro_f1": vs["action"]["macro_f1"],
            "val_seen_xview_top1": vs["xview"]["top1"],
            "val_seen_xview_top5": vs["xview"]["top5"],
            "val_seen_r1": vs["retrieval"]["recall@1"],
            "val_seen_r5": vs["retrieval"]["recall@5"],
            "val_seen_r10": vs["retrieval"]["recall@10"],
            "val_unseen_action_acc": vu["action"]["accuracy"],
            "val_unseen_macro_f1": vu["action"]["macro_f1"],
            "val_unseen_xview_top1": vu["xview"]["top1"],
            "val_unseen_xview_top5": vu["xview"]["top5"],
            "val_unseen_r1": vu["retrieval"]["recall@1"],
            "val_unseen_r5": vu["retrieval"]["recall@5"],
            "val_unseen_r10": vu["retrieval"]["recall@10"],
            "per_class": _prf(conf),
            "left_right": {
                "pred_TurnRight": int(conf[:, 2].sum()),
                "pred_TurnLeft": int(conf[:, 3].sum()),
                "true_TurnRight_correct": int(conf[2, 2]),
                "true_TurnLeft_correct": int(conf[3, 3]),
                "TurnRight_as_TurnLeft": int(conf[2, 3]),
                "TurnLeft_as_TurnRight": int(conf[3, 2]),
                "support_TurnRight": int(conf[2, :].sum()),
                "support_TurnLeft": int(conf[3, :].sum()),
            },
        }
        summary.append(entry)
        records[label] = best
        curves[label] = [(int(r["step"]), float(r["val_seen"].get("selection_score", -1)))
                         for r in history]

    if not summary:
        print("no runs")
        return 1
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "results" / "bev_pose_comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    flat = [{k: v for k, v in row.items() if k not in ("per_class", "left_right")}
            for row in summary]
    with (out / "results" / "bev_pose_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)

    labels = [r["run"] for r in summary]
    keys = [("val_seen_action_acc", "VS action acc"), ("val_seen_macro_f1", "VS macro-F1"),
            ("val_seen_xview_top1", "VS xview Top-1"), ("val_seen_r5", "VS R@5"),
            ("val_unseen_action_acc", "VU action acc")]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.4))
    for ax, (key, title) in zip(axes, keys):
        ax.bar(range(len(summary)), [r[key] for r in summary], color="#2f6fdd")
        ax.set_xticks(range(len(summary)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "bev_pose_metrics.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(records), figsize=(4 * len(records), 3.8))
    if len(records) == 1:
        axes = [axes]
    for ax, (label, best) in zip(axes, records.items()):
        conf = np.asarray(best["val_seen"]["action"]["confusion"])
        im = ax.imshow(conf, cmap="Blues")
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(CLASSES, fontsize=7)
        ax.set_title(label, fontsize=9)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, int(conf[i, j]), ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=axes, shrink=0.7)
    fig.suptitle("Val-Seen confusion matrices (best checkpoint)")
    fig.savefig(out / "figures" / "bev_pose_confusions.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, pts in curves.items():
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=label)
    ax.set_xlabel("step"); ax.set_ylabel("Val-Seen selection score")
    ax.set_title("Checkpoint selection curves"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "bev_pose_curves.png", dpi=220)
    plt.close(fig)

    print(f"wrote {len(summary)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
