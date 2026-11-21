#!/usr/bin/env python3
"""Stage 15 Phase B analysis: action agreement / disagreement on shared states."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)
CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]
MODELS = ["small", "medium", "ts", "tm", "large"]
NAMES = {"small": "Small", "medium": "Medium", "ts": "TS", "tm": "TM", "large": "Large"}
COLORS = {"small": "#4c78a8", "medium": "#f58518", "ts": "#54a24b",
          "tm": "#b279a2", "large": "#e45756"}


def main() -> None:
    df = pd.read_csv(HERE / "expert_action_predictions.csv")
    piv = df.pivot(index="state_index", columns="model", values="action").dropna()
    meta = df.drop_duplicates("state_index").set_index("state_index")[
        ["label", "label_name"]]
    piv = piv.join(meta)
    n = len(piv)

    # per-model accuracy and confidence
    acc_rows = []
    for m in MODELS:
        sub = df[df.model == m]
        acc = float((sub.action == sub.label).mean())
        acc_rows.append({
            "model": m, "model_name": NAMES[m], "accuracy": acc,
            "mean_confidence": float(sub.confidence.mean()),
            "mean_entropy": float(sub.entropy.mean()),
            "high_conf_wrong": int(((sub.confidence > 0.8) &
                                    (sub.action != sub.label)).sum()),
            "high_conf_wrong_pct": float(((sub.confidence > 0.8) &
                                          (sub.action != sub.label)).mean()),
        })
    acc_df = pd.DataFrame(acc_rows)
    acc_df.to_csv(HERE / "expert_action_accuracy.csv", index=False)

    # agreement: all-agree, pairwise, by teacher label
    agree = piv[MODELS].nunique(axis=1) == 1
    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        same = (piv[a] == piv[b])
        rows.append({"model_a": a, "model_b": b, "agreement": float(same.mean()),
                     "n_agree": int(same.sum()), "n_disagree": int((~same).sum())})
    pair_df = pd.DataFrame(rows)
    pair_df.to_csv(HERE / "expert_action_agreement.csv", index=False)

    # disagreement by teacher label
    by_label = []
    for lab, name in enumerate(CLASSES):
        sub = piv[piv.label == lab]
        if len(sub) == 0:
            continue
        for m in MODELS:
            by_label.append({
                "label": lab, "label_name": name, "model": m,
                "n_states": len(sub),
                "action_match_teacher": float((sub[m] == lab).mean()),
                "all_agree_rate": float((sub[MODELS].nunique(axis=1) == 1).mean()),
            })
    by_label_df = pd.DataFrame(by_label)
    by_label_df.to_csv(HERE / "expert_disagreement_by_state.csv", index=False)

    # Stop-specific
    stop_sub = piv[piv.label == 0]
    stop_rows = []
    for m in MODELS:
        stop_rows.append({
            "model": m,
            "stop_pred_rate_on_stop_states": float((stop_sub[m] == 0).mean()),
            "false_stop_rate_on_nonstop": float(
                (piv[piv.label != 0][m] == 0).mean()),
        })
    stop_df = pd.DataFrame(stop_rows)
    stop_df.to_csv(HERE / "stop_decision_analysis.csv", index=False)

    # which model is correct on disagreement states
    dis = piv[~agree]
    winner = []
    for _, r in dis.iterrows():
        correct = [m for m in MODELS if r[m] == r.label]
        winner.append({"state_index": int(r.name), "label": int(r.label),
                       "n_correct": len(correct),
                       "correct_models": "+".join(correct) if correct else "none"})
    win_df = pd.DataFrame(winner)
    win_df.to_csv(HERE / "disagreement_winners.csv", index=False)

    # ---------- figures ----------
    # confusion matrices
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    for ax, m in zip(axes, MODELS):
        sub = df[df.model == m]
        cm = np.zeros((4, 4), int)
        for _, r in sub.iterrows():
            cm[int(r.label), int(r.action)] += 1
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(CLASSES, rotation=45, fontsize=7)
        ax.set_yticklabels(CLASSES, fontsize=7)
        ax.set_title(NAMES[m], fontsize=9)
        for i in range(4):
            for j in range(4):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > cm.max() * 0.55 else "black")
    fig.suptitle("Phase B: expert confusion matrices on shared states")
    fig.tight_layout()
    fig.savefig(FIG / "phaseB_confusion_matrices.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # pairwise agreement heatmap
    hm = np.ones((5, 5))
    for r in pair_df.itertuples():
        i, j = MODELS.index(r.model_a), MODELS.index(r.model_b)
        hm[i, j] = hm[j, i] = r.agreement
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(hm, cmap="viridis", vmin=0.6, vmax=1.0)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([NAMES[m] for m in MODELS])
    ax.set_yticklabels([NAMES[m] for m in MODELS])
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{hm[i, j]:.3f}", ha="center", va="center",
                    color="white" if hm[i, j] < 0.85 else "black", fontsize=9)
    fig.colorbar(im, label="action agreement")
    ax.set_title("Pairwise action agreement (1000 shared states)")
    fig.savefig(FIG / "phaseB_pairwise_agreement.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # disagreement rate by teacher label
    fig, ax = plt.subplots(figsize=(6, 4))
    rates = []
    for lab, name in enumerate(CLASSES):
        sub = piv[piv.label == lab]
        rates.append(1 - float((sub[MODELS].nunique(axis=1) == 1).mean()) if len(sub) else np.nan)
    ax.bar(CLASSES, rates, color="#4c78a8")
    for i, v in enumerate(rates):
        if not np.isnan(v):
            ax.text(i, v, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("fraction of states with any disagreement")
    ax.set_title("Expert disagreement by teacher action")
    ax.grid(alpha=.3, axis="y")
    fig.savefig(FIG / "phaseB_disagreement_by_label.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # calibration: confidence vs correctness
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in MODELS:
        sub = df[df.model == m]
        bins = np.linspace(0, 1, 11)
        mids, accs = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = sub[(sub.confidence >= lo) & (sub.confidence < hi)]
            if len(s) >= 5:
                mids.append((lo + hi) / 2)
                accs.append(float((s.action == s.label).mean()))
        ax.plot(mids, accs, "o-", ms=4, color=COLORS[m], label=NAMES[m])
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    ax.set_xlabel("predicted confidence"); ax.set_ylabel("accuracy")
    ax.set_title("Confidence calibration on shared states")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.savefig(FIG / "phaseB_calibration.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---------- report ----------
    lines = [
        "# Stage 15 · 阶段 B：共同状态下的专家行为互补性",
        "",
        f"- 共同状态数：**{n}**（按教师动作分层采样，Stop 150 / Forward 450 / "
        f"TurnLeft 200 / TurnRight 200）",
        "- 五个冻结专家，同一输入，batch=1，顺序加载（无显存冲突）",
        "- Small/Medium 使用原始预处理；TS/TM/Large 使用 Stage 14-R2 归一化",
        "",
        "## B1. 单模型动作准确率与置信度",
        "",
        "| 模型 | 教师动作准确率 | 平均置信度 | 平均熵 | 高置信错误 (>0.8) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in acc_df.itertuples():
        lines.append(f"| {NAMES[r.model]} | {r.accuracy*100:.1f}% | "
                     f"{r.mean_confidence:.3f} | {r.mean_entropy:.3f} | "
                     f"{r.high_conf_wrong} ({r.high_conf_wrong_pct*100:.1f}%) |")
    lines += [
        "",
        "## B2. 专家间动作一致性",
        "",
        f"- 五专家完全一致：**{int(agree.sum())} / {n} = {agree.mean()*100:.1f}%**",
        f"- 存在分歧：**{int((~agree).sum())} / {n} = {(~agree).mean()*100:.1f}%**",
        "",
        "| 组合 | 一致率 | 分歧数 |",
        "|---|---:|---:|",
    ]
    for r in pair_df.sort_values("agreement").itertuples():
        lines.append(f"| {NAMES[r.model_a]} vs {NAMES[r.model_b]} | "
                     f"{r.agreement*100:.1f}% | {r.n_disagree} |")
    lines += [
        "",
        "## B3. 按教师动作的分歧",
        "",
    ]
    for lab, name in enumerate(CLASSES):
        sub = piv[piv.label == lab]
        if len(sub) == 0:
            continue
        d = 1 - float((sub[MODELS].nunique(axis=1) == 1).mean())
        lines.append(f"- **{name}**（{len(sub)} states）：分歧率 {d*100:.1f}%")
    lines += [
        "",
        "## B4. Stop 决策",
        "",
        "| 模型 | Stop 状态上预测 Stop | 非 Stop 状态上误报 Stop |",
        "|---|---:|---:|",
    ]
    for r in stop_df.itertuples():
        lines.append(f"| {NAMES[r.model]} | "
                     f"{r.stop_pred_rate_on_stop_states*100:.1f}% | "
                     f"{r.false_stop_rate_on_nonstop*100:.1f}% |")
    lines += [
        "",
        "## B5. 分歧状态中谁是正确的",
        "",
        "| 正确专家数 | 状态数 |",
        "|---|---:|",
    ]
    for k, v in win_df["n_correct"].value_counts().sort_index().items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## 产物", "",
              "- `shared_states_manifest.csv`", "- `expert_action_predictions.csv`",
              "- `expert_action_agreement.csv`", "- `expert_disagreement_by_state.csv`",
              "- `expert_action_accuracy.csv`", "- `stop_decision_analysis.csv`",
              "- `disagreement_winners.csv`", "- 图：`figures/phaseB_*.png`", ""]
    (HERE / "STATE_LEVEL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
