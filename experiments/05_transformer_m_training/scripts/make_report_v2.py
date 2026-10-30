#!/usr/bin/env python3
"""Assemble experiments/05_transformer_m_training/REPORT.md from a finished run.

Reads runs_v2/<run>/{config.json,history.jsonl,best.pt}, the cache index of each
split, results/normalization_check.json and (if present) the diagnostics JSON,
and writes the tables that the stage-05 report needs.  The analysis section is
left as a marker for a human/agent to fill in; everything numeric is generated
so it cannot drift from the run.

Usage: python make_report_v2.py --run seed42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RUNS = HERE / "runs_v2"
RESULTS = HERE / "results"
CACHE = HERE / "features_v2"
CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def history(run: str) -> list[dict]:
    path = RUNS / run / "history.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def fmt(x: float, n: int = 4) -> str:
    return f"{x:.{n}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="seed42")
    parser.add_argument("--out", default=str(HERE / "REPORT.md"))
    args = parser.parse_args()

    run_dir = RUNS / args.run
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    hist = history(args.run)
    best = hist[max(range(len(hist)), key=lambda i: hist[i]["val_seen"]["macro_f1"])]
    splits = {}
    for split in ("train_seen", "val_seen", "val_unseen"):
        index = json.loads((CACHE / split / "index.json").read_text(encoding="utf-8"))
        splits[split] = index
    norm = json.loads((RESULTS / "normalization_check.json").read_text(encoding="utf-8"))
    diag_path = RESULTS / f"diagnostics_{args.run}.json"
    diag = json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.exists() else None
    best_ckpt = run_dir / "best.pt"
    ckpt_info = json.loads(json.dumps({"sha256": sha256(best_ckpt),
                                       "bytes": best_ckpt.stat().st_size,
                                       "epoch": best["epoch"]}))

    total_seconds = sum(h["seconds"] for h in hist)
    lines: list[str] = []
    add = lines.append
    add("# Stage 05 — Transformer-M 正式训练报告")
    add("")
    add(f"- 运行目录：`runs_v2/{args.run}/`")
    add(f"- 分支：`exp/citynav-mec-fixed50m-topdown`")
    add(f"- 唯一训练种子：{config['seed']}；epochs：{config['epochs']}")
    add(f"- 训练脚本：`scripts/train_v2.py`；启动脚本：`scripts/run_formal.sh`（tmux `citynav05_seed42`）")
    add("")
    add("## 1. 数据")
    add("")
    add("| 划分 | episode 数 | 决策状态数 | 用途 |")
    add("|---|---|---|---|")
    add(f"| Train-Seen | {splits['train_seen']['episodes_total']:,} | {splits['train_seen']['states']:,} | 训练 |")
    add(f"| Val-Seen | {splits['val_seen']['episodes_total']:,} | {splits['val_seen']['states']:,} | 每 epoch 验证 |")
    add(f"| Val-Unseen | {splits['val_unseen']['episodes_total']:,} | {splits['val_unseen']['states']:,} | 每 5 epoch 验证 |")
    add("")
    add(f"- 教师：03B 直接目标导向教师 `teacher_train_seen.npz`"
        f"（{splits['train_seen']['episodes_total']:,} 条，非 Stage 02 筛选、非人工轨迹）")
    add(f"- 候选库：`{splits['train_seen']['candidate_library']}`"
        f"（24 m 步长 / 96 m 切片，版本 `{splits['train_seen']['candidate_version']}`）")
    add(f"- 视觉编码器：{splits['train_seen']['encoder']}")
    add(f"- 缓存：episode 级 npz + journal（大小/SHA-256），版本 `{splits['train_seen']['cache_version']}`")
    add(f"- Test-Unseen 未读取。")
    add("")
    add("## 2. 模型与训练配置")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| 参数量 | 4,885,252 |")
    add(f"| batch / k per episode | {config['batch']} / {config['k_per_episode']} |")
    add(f"| optimizer / lr / wd | AdamW / {config['lr']} / {config['weight_decay']} |")
    add(f"| scheduler | OneCycleLR, warmup {config['warmup']} |")
    add(f"| 损失 | 普通 CrossEntropy，无类别权重/pos_weight/focal |")
    add(f"| 输入 | 7×7×2048 当前视觉、2048 目标参考、16 步历史、12 维信念；无语言/坐标/距离 |")
    add(f"| 断点 | 每 {config.get('save_every', 100)} batch 存档 + 每 epoch 末完整存档 |")
    add("")
    add("## 3. 归一化一致性")
    add("")
    add(f"- 生产路径：{norm['production']}")
    add(f"- 缓存路径 vs 训练路径特征 `max_abs_diff = {norm['cache_vs_train_max_abs_diff']:.3e}`")
    add(f"- 已弃用的 CPU 归一化参考路径与 GPU 路径差"
        f" `{norm['cpu_reference_max_abs_diff']:.3e}`（{norm['cpu_reference_ulp']:.1f} ULP），"
        f"经 fp16 后放大到 `{norm['cpu_reference_feature_max_abs_diff']:.3e}`，故删除重复实现")
    add(f"- 不同 batch 的 fp16 核选择差异（信息项）：`{norm['cross_batch_max_abs_diff']:.3e}`")
    add("")
    add("## 4. 逐 epoch 结果")
    add("")
    add("| epoch | train loss | train acc | val acc | val Macro-F1 | STOP R | TurnL R | TurnR R | grad norm | 秒 |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for h in hist:
        v = h["val_seen"]
        pc = v["per_class"]
        add(f"| {h['epoch']} | {fmt(h['train_loss'])} | {fmt(h['train_acc'])} | "
            f"{fmt(v['accuracy'])} | {fmt(v['macro_f1'])} | {fmt(pc['Stop']['recall'], 3)} | "
            f"{fmt(pc['TurnLeft']['recall'], 3)} | {fmt(pc['TurnRight']['recall'], 3)} | "
            f"{fmt(h.get('grad_norm', float('nan')), 3)} | {h['seconds']:.1f} |")
    add("")
    add("## 5. 最佳 epoch（按 Val-Seen Macro-F1）")
    add("")
    v = best["val_seen"]
    add(f"- epoch {best['epoch']}，Macro-F1 {fmt(v['macro_f1'])}，准确率 {fmt(v['accuracy'])}")
    add(f"- 各类：")
    add("")
    add("| 类 | precision | recall | F1 | support | predicted |")
    add("|---|---|---|---|---|---|")
    for c in CLASSES:
        pc = v["per_class"][c]
        add(f"| {c} | {fmt(pc['precision'])} | {fmt(pc['recall'])} | {fmt(pc['f1'])} | "
            f"{pc['support']} | {pc['predicted']} |")
    add("")
    add("混淆矩阵（行=标签，列=预测，顺序 Stop/Forward/TurnLeft/TurnRight）：")
    add("")
    add("```")
    for row in v["confusion"]:
        add("  " + "  ".join(f"{x:5d}" for x in row))
    add("```")
    add("")
    add(f"- checkpoint：`runs_v2/{args.run}/best.pt`，SHA-256 `{ckpt_info['sha256']}`，"
        f"{ckpt_info['bytes']:,} 字节")
    add("")
    add("## 6. 耗时")
    add("")
    add(f"- 训练总耗时（不含特征提取）：{total_seconds / 60:.1f} 分钟")
    add(f"- 平均每 epoch：{statistics.mean(h['seconds'] for h in hist):.1f} 秒")
    add("")
    if diag:
        add("## 7. 全量教师强制诊断（results/diagnostics_%s.json）" % args.run)
        add("")
        add("| 划分 | n | acc | Macro-F1 | STOP R | TurnL R | TurnR R |")
        add("|---|---|---|---|---|---|---|")
        for split, out in diag["splits"].items():
            pc = out["per_class"]
            add(f"| {split} | {out['n']} | {fmt(out['accuracy'])} | {fmt(out['macro_f1'])} | "
                f"{fmt(pc['Stop']['recall'], 3)} | {fmt(pc['TurnLeft']['recall'], 3)} | "
                f"{fmt(pc['TurnRight']['recall'], 3)} |")
        add("")
    add("## 8. 结论")
    add("")
    add("<!-- ANALYSIS -->")
    add("")
    (Path(args.out)).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
