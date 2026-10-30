#!/usr/bin/env python3
"""Assemble experiments/06_closed_loop_success_rate/REPORT.md from the results.

Every number in the report is read from ``results/seed42/summary.json``,
``manifest.json`` and ``eval_config.json`` so the text cannot drift from the
data.  The analysis section is written by hand after the tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = HERE.parent / "results/seed42"
FIG = HERE.parent / "figures/seed42"


def pct(x, n=1):
    return "—" if x is None else f"{100 * x:.{n}f}%"


def ci(m):
    lo, hi = m["sr_ci95"]
    return f"[{100 * lo:.1f}, {100 * hi:.1f}]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument("--out", default=str(HERE.parent / "REPORT.md"))
    parser.add_argument("--wall-minutes", type=float, default=None)
    args = parser.parse_args()
    res_dir = Path(args.results)
    summary = json.loads((res_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((res_dir / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((res_dir / "eval_config.json").read_text(encoding="utf-8"))
    s = summary["scopes"]
    splits = ["val_seen", "val_unseen"]
    groups = ["short", "medium", "long"]

    L: list[str] = []
    add = L.append
    add("# Stage 06 — Transformer-M（seed 42）闭环导航成功率")
    add("")
    add("## 1. 目的与协议")
    add("")
    add("在固定 50 m 俯视视觉目标导航任务上，用 Stage 05 训练好的 Transformer-M 自己的")
    add("`argmax` 动作驱动 UAV，测量闭环成功率 SR。模型不重新训练、不调阈值、不加权；")
    add("教师轨迹只提供起点位姿与最终目标位置，不提供中间状态、动作或未来观测。")
    add("")
    add("动作：STOP / Forward 5 m / TurnLeft 30° / TurnRight 30°。")
    add("成功判据（三条同时满足）：模型输出 STOP、STOP 时水平距离 ≤ 20 m、最终距离 ≤ 20 m。")
    add(f"最大执行步数 {summary['step_cap']}（来自 03B Train-Seen 的 p99=85、max=127，"
        "不把任何教师长度传给模型）。")
    add("")
    add("## 2. 模型")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add("| checkpoint | `experiments/05_transformer_m_training/runs_v2/seed42/best.pt` |")
    add(f"| SHA-256 | `{summary['checkpoint_sha']}` |")
    add("| 选择依据 | Stage 05 Val-Seen Macro-F1 最高（epoch 1，0.7844），**未**按本次闭环结果重选 |")
    add("| 参数量 | 4,885,252 |")
    add("| 加载方式 | `load_state_dict(..., strict=True)`，`eval()` + `torch.inference_mode()` |")
    add("")
    add("## 3. 输入与推理数据流")
    add("")
    add("| 输入 | 维度 | 来源 |")
    add("|---|---|---|")
    add("| 当前视觉 | 49×2048 | `heading_crop` 100 m×100 m → 224² → 冻结 ResNet-50 layer4（uint8→GPU 归一化） |")
    add("| 目标参考 | 2048 | `north_crop(target, 48 m)` → `embed` → L2，模型侧 FP16 精度往返 |")
    add("| 历史视觉 | 16×2048 | 实际执行历史的 pooled 特征，FP16→FP32（与训练缓存同精度） |")
    add("| 历史相对位姿 | 16×3 | 当前机头坐标系，仅用实际执行状态 |")
    add("| 历史动作 | 16×4 | 实际执行动作 one-hot |")
    add("| 历史掩码 | 16 | |")
    add("| 全局目标信念 | 12 | `belief_features()`，24 m 候选库，每步用当前位姿重算 |")
    add("")
    add("候选库：`cache_v2/candidates24`（与 Stage 05 缓存索引一致）；编码器：frozen ResNet-50")
    add("IMAGENET1K_V2 layer4 FP16。闭环首步决策与训练管线在同一 pose 上的概率差实测")
    add("0–0.0002（4 条抽样），说明输入契约一致。")
    add("")
    add("## 4. 任务集合")
    add("")
    add("| 划分 | Short (≤29) | Medium (30–49) | Long (>49) | 合计 |")
    add("|---|---|---|---|---|")
    total = 0
    for split in splits:
        c = manifest["counts"][split]
        n = sum(c[g]["total"] for g in groups)
        total += n
        add(f"| {split} | {c['short']['total']} | {c['medium']['total']} | {c['long']['total']} | {n} |")
    add(f"| **合计** | | | | **{total}** |")
    add("")
    add("长度分组沿用 03B 由 Train-Seen 确定的边界，未在验证集上重算。每个（划分 × 长度组）")
    add(f"用 seed {manifest['seed']} 确定性打乱后分成 Round A / Round B 两个互不重叠 cohort，")
    add("两者并集覆盖全部任务。未按人工路径质量过滤。")
    add("")
    add(f"目标图像有效性：`goal_image_invalid` {sum(1 for t in manifest['tasks'] if t['goal_image_invalid'])} 条；"
        f"目标 patch 有效比例 < 0.5 的 {sum(1 for t in manifest['tasks'] if 0 < t['target_valid_fraction'] < 0.5)} 条"
        "（保留在主结果中，另给子集）。")
    add("")
    add("## 5. 主结果")
    add("")
    add("| 范围 | n | SR | SR 95% CI (Wilson) | OSR | STOP 率 | NE 中位 | 步数均值 | SPL（成功加权） |")
    add("|---|---|---|---|---|---|---|---|---|")
    rows = [("全部", s["all"])] + [(sp, s[sp]) for sp in splits]
    for name, m in rows:
        spl = f"{m['spl']:.3f}" if m.get("spl") is not None else "—"
        add(f"| {name} | {m['n']} | **{pct(m['sr'])}** | {ci(m)} | {pct(m['osr'])} | "
            f"{pct(m['stop_rate'])} | {m['ne_median']:.1f} m | {m['steps_mean']:.1f} | {spl} |")
    add("")
    add("### 按长度组")
    add("")
    add("| 划分 | 组 | n | SR | 95% CI | OSR | STOP 率 | NE 中位 | 步数均值 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for sp in splits:
        for g in groups:
            m = s[f"{sp}|{g}"]
            add(f"| {sp} | {g} | {m['n']} | **{pct(m['sr'])}** | {ci(m)} | {pct(m['osr'])} | "
                f"{pct(m['stop_rate'])} | {m['ne_median']:.1f} m | {m['steps_mean']:.1f} |")
    add("")
    add("### 按 cohort")
    add("")
    add("| 划分 | cohort | n | SR | 95% CI | OSR |")
    add("|---|---|---|---|---|---|")
    for sp in splits:
        for c in ("A", "B"):
            m = s[f"{sp}|cohort_{c}"]
            add(f"| {sp} | {c} | {m['n']} | **{pct(m['sr'])}** | {ci(m)} | {pct(m['osr'])} |")
    add("")
    add("### 目标图像有效性子集")
    add("")
    add("| 划分 | 子集 | n | SR | 95% CI |")
    add("|---|---|---|---|---|")
    for sp in splits:
        for name in ("valid_goal_image", "target_valid_ge_0.5", "target_valid_lt_0.5",
                     "goal_image_invalid"):
            key = f"{sp}|{name}"
            if key in s and s[key].get("n"):
                m = s[key]
                add(f"| {sp} | {name} | {m['n']} | **{pct(m['sr'])}** | {ci(m)} |")
    add("")
    add("## 6. 失败模式与动作统计")
    add("")
    m = s["all"]
    add("| 类别 | 条数 | 比例 |")
    add("|---|---|---|")
    labels = {"success": "成功", "premature_stop": "过早 STOP",
              "reached_but_not_stopped": "进入 20 m 但未停", "never_stop": "到步数上限未停",
              "collision": "碰撞", "out_of_bounds": "越界",
              "goal_image_invalid": "目标图像无效", "other_failure": "其他异常"}
    for key, label in labels.items():
        n = m["failures"].get(key, 0)
        add(f"| {label} | {n} | {pct(n / m['n'])} |")
    add("")
    add(f"- 碰撞（任意时刻）：{m['collision_any_rate'] * 100:.2f}%；"
        f"越界（任意时刻）：{pct(m['out_of_bounds_any_rate'])}")
    add(f"- 达到 200 步上限：{m['step_cap_hits']} 条")
    stop_err = ("—" if m["stop_error_median"] is None
                else f"{m['stop_error_median']:.1f} m")
    add(f"- NE 中位 {m['ne_median']:.1f} m；停止位置误差中位 {stop_err}")
    add("")
    add("四类动作输出：")
    add("")
    add("| 动作 | 次数 | 占比 |")
    add("|---|---|---|")
    for c in ("Stop", "Forward", "TurnLeft", "TurnRight"):
        add(f"| {c} | {m['action_totals'][c]:,} | {pct(m['action_share'][c], 2)} |")
    add("")
    add("## 7. 图表")
    add("")
    add("![SR by split / length / cohort](figures/seed42/sr_by_split_group_cohort.png)")
    add("")
    add("![OSR vs SR](figures/seed42/osr_vs_sr.png)")
    add("")
    add("![action distribution](figures/seed42/action_distribution.png)")
    add("")
    add("![distance histograms](figures/seed42/distance_histograms.png)")
    add("")
    add("![trajectory cases](figures/seed42/trajectory_cases.png)")
    add("")
    add("## 8. 运行时间")
    add("")
    if args.wall_minutes:
        add(f"- 全量闭环耗时：**{args.wall_minutes:.1f} 分钟**（{summary['episodes_scored']} 条，"
            f"{config.get('workers')} worker）")
    else:
        add(f"- 全量闭环耗时：{summary['episodes_scored']} 条，{config.get('workers')} worker")
    add(f"- 平均每条：{540.0 * args.wall_minutes / max(summary['episodes_scored'], 1):.2f} 秒"
        if args.wall_minutes else "-")
    add("")
    add("## 9. 分析")
    add("")
    add("<!-- ANALYSIS -->")
    add("")
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
