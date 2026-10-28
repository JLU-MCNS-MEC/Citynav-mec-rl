"""Aggregate E1-E4 runs into one comparison table and figure.

Reads ``runs/visual_encoder_citynav/<run>/val_history.json`` (written by
train.py) and produces:

* ``results/ablation_comparison_20261028.csv`` / ``.json``
* ``figures/ablation_validation_20261028.png`` (Val-Seen metrics per run)
* ``figures/ablation_curves_20261028.png`` (selection score vs step)

The old baseline (seed 0, different initial encoder) is included for reference
only and is marked ``reference_only``.
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
DEFAULT_RUNS = {
    "E1": "abl_20261028_e1_seed42",
    "E2": "abl_20261028_e2_seed42",
    "E3": "abl_20261028_e3_seed42",
    "E4": "abl_20261028_e4_seed42",
}
BASELINE = (
    PROJECT_ROOT
    / "experiments/visual_encoder_citynav/results/full_20261027_masks_b8_val_history.json"
)


def _score(record: dict, split: str) -> float:
    return float(record[split].get("selection_score", -1.0))


def _row(name: str, history: list[dict], reference_only: bool) -> dict:
    best = max(history, key=lambda r: _score(r, "val_seen"))
    vs, vu = best["val_seen"], best["val_unseen"]
    return {
        "run": name,
        "reference_only": reference_only,
        "best_step": int(best["step"]),
        "val_seen_selection_score": _score(best, "val_seen"),
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
        "per_class_val_seen": vs["action"].get("per_class_accuracy", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()

    rows = []
    curves = {}
    for name, run in DEFAULT_RUNS.items():
        path = PROJECT_ROOT / "runs" / "visual_encoder_citynav" / run / "val_history.json"
        if not path.exists():
            print(f"missing {name}: {path}")
            continue
        history = json.loads(path.read_text())
        rows.append(_row(name, history, False))
        curves[name] = [(r["step"], _score(r, "val_seen")) for r in history]

    if BASELINE.exists():
        history = json.loads(BASELINE.read_text())
        rows.append(_row("baseline_full_20261027_masks_b8", history, True))

    if not rows:
        print("no runs found")
        return 1

    out = args.out_dir
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    json_path = out / "results" / "ablation_comparison_20261028.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    csv_path = out / "results" / "ablation_comparison_20261028.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[k for k in rows[0] if k != "per_class_val_seen"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({k: v for k, v in r.items() if k != "per_class_val_seen"})

    # --- figure 1: Val-Seen metric bars -----------------------------------
    metrics = [
        ("val_seen_macro_f1", "macro-F1"),
        ("val_seen_xview_top1", "xview Top-1"),
        ("val_seen_r5", "R@5"),
        ("val_seen_action_acc", "action acc"),
    ]
    names = [r["run"] for r in rows]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.4))
    for ax, (key, label) in zip(axes, metrics):
        values = [r[key] for r in rows]
        colors = ["#c8c8c8" if r["reference_only"] else "#2f6fdd" for r in rows]
        ax.bar(range(len(rows)), values, color=colors)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("CityNav visual-encoder ablations — Val-Seen", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "figures" / "ablation_validation_20261028.png", dpi=220)
    plt.close(fig)

    # --- figure 2: selection-score curves ---------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, pts in curves.items():
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=name)
    ax.set_xlabel("update step")
    ax.set_ylabel("Val-Seen selection score")
    ax.set_title("Checkpoint selection curves")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "ablation_curves_20261028.png", dpi=220)
    plt.close(fig)

    print(f"wrote {len(rows)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
