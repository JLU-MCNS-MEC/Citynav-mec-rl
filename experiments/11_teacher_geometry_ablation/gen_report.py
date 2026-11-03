#!/usr/bin/env python3
"""阶段报告生成器：从训练 history.jsonl + 闭环 journal 生成中文 REPORT.md。

用法: python gen_report.py <model> <run_dir> <eval_out_dir>
示例: python gen_report.py small \
    experiments/11_teacher_geometry_ablation/11B_transformer_m_v4/results/runs_small/seed42 \
    experiments/v4_closed_loop_revalidation/results/small_val_seen_20261103
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]


def load_history(run_dir: Path) -> list[dict]:
    rows = []
    hp = run_dir / "history.jsonl"
    if hp.exists():
        for line in hp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_journal(out: Path) -> list[dict]:
    recs = []
    for p in sorted(out.glob("journal*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def closed_loop_metrics(recs: list[dict]) -> dict:
    n = len(recs)
    succ = [r for r in recs if bool(r.get("success"))]
    osr = [r for r in recs if bool(r.get("oracle_success"))]
    ne = [float(r.get("final_distance", float("nan"))) for r in recs]
    ne = [v for v in ne if v == v]
    spl = [float(r.get("spl", float("nan"))) for r in recs]
    spl = [v for v in spl if v == v]
    steps = [int(r.get("step_count", 0)) for r in recs]
    stop_reasons: dict[str, int] = {}
    for r in recs:
        sr = str(r.get("stop_reason", "?"))
        stop_reasons[sr] = stop_reasons.get(sr, 0) + 1
    return {
        "n": n, "sr": len(succ) / max(n, 1), "osr": len(osr) / max(n, 1),
        "ne_mean": float(np.mean(ne)) if ne else float("nan"),
        "spl_mean": float(np.mean(spl)) if spl else float("nan"),
        "steps_mean": float(np.mean(steps)) if steps else float("nan"),
        "stop_reasons": stop_reasons,
    }


def main() -> int:
    model, run_dir_s, eval_s = sys.argv[1], sys.argv[2], sys.argv[3]
    run_dir, out = Path(run_dir_s), Path(eval_s)
    hist = load_history(run_dir)
    recs = load_journal(out)
    m = closed_loop_metrics(recs)

    lines = [f"# {model.upper()} 阶段报告（{Path(run_dir_s).name}）", ""]
    if hist:
        best_i = int(np.argmax([h["val_seen"]["macro_f1"] for h in hist]))
        best = hist[best_i]
        lines += [
            f"- 训练：{len(hist)} epochs 完成；best = epoch {best['epoch']}"
            f"（Val Macro-F1 {best['val_seen']['macro_f1']:.4f}）",
            f"- 训练耗时：{sum(h['seconds'] for h in hist) / 3600:.1f} h；"
            f"最终 LR {best['lr']:.2e}；平均梯度范数 {best['grad_norm']:.2f}",
            f"- 各 epoch 详见 `{run_dir_s}/history.jsonl`",
            "",
        ]
        lines.append("## 各 Epoch 摘要")
        lines.append("| epoch | train_loss | train_acc | val_loss | val_acc | macro_f1 | stop_recall | turn_recall | lr | sec |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for h in hist:
            v = h["val_seen"]
            tr = (v["per_class"]["TurnLeft"]["recall"] + v["per_class"]["TurnRight"]["recall"]) / 2
            lines.append(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['train_acc']:.4f} | "
                         f"{v['loss']:.4f} | {v['accuracy']:.4f} | {v['macro_f1']:.4f} | "
                         f"{v['per_class']['Stop']['recall']:.3f} | {tr:.3f} | "
                         f"{h['lr']:.2e} | {h['seconds']:.0f} |")
        lines += ["", ]
    lines += [
        "## Val-Seen 闭环（修正协议：STOP 20m / 200 步 / 2,498 任务）",
        f"- 任务数：{m['n']}；SR **{m['sr'] * 100:.2f}%**；OSR {m['osr'] * 100:.2f}%",
        f"- NE 均值 {m['ne_mean']:.1f} m；SPL 均值 {m['spl_mean']:.3f}；平均步数 {m['steps_mean']:.1f}",
        f"- 失败分类（stop_reason）：{json.dumps(m['stop_reasons'], ensure_ascii=False)}",
        "",
        f"- 评测证据：`{eval_s}/`（journal + eval_config.json + episodes/）",
    ]
    out_md = Path(run_dir_s) / "REPORT.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
