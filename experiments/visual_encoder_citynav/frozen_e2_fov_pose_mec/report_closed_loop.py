#!/usr/bin/env python3
"""Collect closed-loop summary JSONs into one Chinese markdown + CSV report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SPLITS = ("val_seen", "val_unseen")
MODELS = ("small", "transformer_m")
VARIANTS = ("full", "map_free")


def _load(out_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(out_dir.glob("*_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        rows.append(summary)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "closed_loop/formal",
    )
    args = parser.parse_args()

    rows = _load(args.out_dir)
    if not rows:
        raise SystemExit(f"no summary JSON in {args.out_dir}")

    rows.sort(key=lambda r: (SPLITS.index(r["split"]), MODELS.index(r["model"]),
                             VARIANTS.index(r.get("variant", "full"))))

    csv_path = args.out_dir / "closed_loop_comparison.csv"
    fields = [
        "split", "model", "variant", "episodes", "success_rate",
        "oracle_success_rate", "spl", "ne_m", "ndtw", "mean_steps",
        "stop_rate", "mean_step_latency_s", "shards",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row.setdefault("variant", "full")
            writer.writerow(row)

    lines = [
        "# 实验 A 闭环导航评估（冻结 E2-FOV-Pose + MEC）",
        "",
        "- 冻结编码器 SHA-256：`441de539f1d30f2be68d77a1c954daef5f3ab3bd3501f33b2c9798f89517011b`",
        "- 决策：MEC 六类 logits 的 argmax（与 token 验证一致）",
        "- 成功：终止位置距目标 ≤ 20 m；oracle 成功：rollout 中最小距离 ≤ 20 m",
        "- 每划分 24 条、最多 100 步、8 分片；样本量小，置信区间宽，不作显著性声明",
        "- Test-Unseen 未读取",
        "",
        "| split | model | variant | n | SR | oracle SR | SPL | NE (m) | nDTW | steps | stop rate | s/step |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['model']} | {row.get('variant', 'full')} | "
            f"{row['episodes']} | {row['success_rate']:.3f} | "
            f"{row['oracle_success_rate']:.3f} | {row['spl']:.3f} | "
            f"{row['ne_m']:.1f} | {row['ndtw']:.3f} | {row['mean_steps']:.1f} | "
            f"{row['stop_rate']:.3f} | {row['mean_step_latency_s']:.2f} |"
        )
    lines += [
        "",
        "## 解读要点",
        "",
        "- `full` 变体在 token 验证集上 Stop 召回为 0，闭环中同样不停止，"
        "因而全部跑满步数；SR 与 oracle SR 的差异说明轨迹是否曾接近目标。",
        "- `map_free` 变体能够 Stop，但转向类动作识别差，路径质量需结合 SPL/nDTW 看。",
        "- 渲染成本约 3–8 s/步（取决于地图密度与并行度），是本次评估规模的主要限制。",
    ]
    (args.out_dir / "CLOSED_LOOP_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for axis, split in zip(axes, SPLITS):
        subset = [r for r in rows if r["split"] == split]
        if not subset:
            axis.set_visible(False)
            continue
        labels = [f"{r['model']}\n{r.get('variant', 'full')}" for r in subset]
        xs = np.arange(len(subset))
        axis.bar(xs - 0.18, [r["success_rate"] for r in subset], width=0.36,
                 label="SR")
        axis.bar(xs + 0.18, [r["oracle_success_rate"] for r in subset],
                 width=0.36, label="oracle SR")
        axis.set_xticks(xs)
        axis.set_xticklabels(labels, fontsize=7)
        axis.set_ylim(0, 1)
        axis.set_ylabel("rate")
        axis.set_title(split)
        axis.grid(alpha=0.25, axis="y")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "closed_loop_rates.png", dpi=200)
    plt.close(fig)

    print(f"wrote {csv_path} and CLOSED_LOOP_REPORT.md ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
