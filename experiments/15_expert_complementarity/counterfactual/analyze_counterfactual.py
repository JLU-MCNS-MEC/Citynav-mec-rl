#!/usr/bin/env python3
"""Analyse the 100-state counterfactual branch results."""
from __future__ import annotations

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
KEYS = ["small", "medium", "ts", "tm", "large"]
NAMES = {"small": "Small", "medium": "Medium", "ts": "TS", "tm": "TM", "large": "Large"}
COLORS = {"small": "#4c78a8", "medium": "#f58518", "ts": "#54a24b",
          "tm": "#b279a2", "large": "#e45756"}


def main() -> None:
    df = pd.read_csv(HERE / "branch_500" / "branch_rollout_results.csv")
    df["success"] = df["success"].astype(str).str.lower().isin(["true", "1"])
    df["state_id"] = df["episode_id"] + "@" + df["start_step"].astype(str)

    # per-expert aggregates
    agg = df.groupby("first_expert").agg(
        rollouts=("success", "size"), sr=("success", "mean"),
        mean_steps=("steps_used", "mean"), mean_final_dist=("final_dist_m", "mean"),
        mean_spl=("spl", "mean"),
        oob=("out_of_bounds", "sum"), collision=("collision", "sum"),
    ).reindex(KEYS)
    agg.to_csv(HERE / "single_step_expert_effect.csv")

    # per-state pivot
    piv = df.pivot(index="state_id", columns="first_expert", values="success")
    piv = piv.reindex(columns=KEYS)
    n_states = len(piv)
    sensitive = piv.nunique(axis=1) > 1
    wins = piv.sum(axis=1)

    # winner sets
    winner_rows = []
    for state, row in piv.iterrows():
        ok = [k for k in KEYS if bool(row.get(k, False))]
        winner_rows.append({"state_id": state, "n_success": len(ok),
                            "winners": "+".join(ok) if ok else "none",
                            "none": len(ok) == 0})
    win_df = pd.DataFrame(winner_rows)
    win_df.to_csv(HERE / "state_winner_sets.csv", index=False)

    # state-level sensitivity
    sens = {
        "n_states": int(n_states),
        "states_with_different_outcome": int(sensitive.sum()),
        "pct_sensitive": float(sensitive.mean()),
        "all_fail_states": int((wins == 0).sum()),
        "all_success_states": int((wins == 5).sum()),
        "small_only": int(((piv.small) & (~piv[["medium", "ts", "tm", "large"]]).all(axis=1)).sum()),
        "ts_only": int(((piv.ts) & (~piv[["small", "medium", "tm", "large"]]).all(axis=1)).sum()),
        "small_success_ts_fail": int((piv.small & ~piv.ts).sum()),
        "ts_success_small_fail": int((piv.ts & ~piv.small).sum()),
    }

    # does static SR ranking predict per-state best?
    static_best = agg["sr"].idxmax()
    sens_states = piv[sensitive]
    sens.update({
        "static_best_expert": static_best,
        "sensitive_states": int(sensitive.sum()),
        "static_best_wins_on_sensitive": int(sens_states[static_best].sum()) if len(sens_states) else 0,
        "pct_static_best_wins_on_sensitive": float(sens_states[static_best].mean()) if len(sens_states) else 0.0,
        "oracle_choice_gain_on_sensitive": float(
            (sens_states.any(axis=1).mean() - sens_states[static_best].mean())
            if len(sens_states) else 0.0),
    })

    # action divergence at the branch step
    traces = json.loads((HERE / "branch_500" / "branch_action_traces.json").read_text())
    first_actions = {}
    for t in traces:
        sid = f"{t['episode_id']}@{t['start_step']}"
        first_actions.setdefault(sid, {})[t["first_expert"]] = t["actions"][0]["action"] if t["actions"] else None
    fa = pd.DataFrame(first_actions).T.reindex(columns=KEYS)
    action_div = (fa.nunique(axis=1) > 1)
    sens["branch_step_action_divergence"] = int(action_div.sum())
    sens["pct_branch_step_action_divergence"] = float(action_div.mean())

    # does branch-step action divergence predict outcome sensitivity?
    both = pd.DataFrame({"action_div": action_div, "outcome_sens": sensitive}).dropna()
    if len(both):
        ct = pd.crosstab(both.action_div, both.outcome_sens)
        sens["crosstab_action_div_vs_outcome_sens"] = ct.to_dict()

    (HERE / "phaseC_summary.json").write_text(json.dumps(sens, indent=2))

    # ---------------- figures ----------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([NAMES[k] for k in KEYS], [agg.loc[k, "sr"] for k in KEYS],
           color=[COLORS[k] for k in KEYS])
    for i, k in enumerate(KEYS):
        ax.text(i, agg.loc[k, "sr"], f"{agg.loc[k,'sr']*100:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("branch success rate"); ax.set_ylim(0, 1)
    ax.set_title(f"Counterfactual SR by first-step expert ({n_states} states)")
    ax.grid(alpha=.3, axis="y")
    fig.savefig(FIG / "phaseC_expert_branch_sr.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(n_states)
    for k in KEYS:
        ax.scatter(xs, [1 if v else 0 for v in piv[k]], s=8, color=COLORS[k],
                   label=NAMES[k], alpha=0.7)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["fail", "success"])
    ax.set_xlabel("state (sorted)"); ax.set_title("Per-state branch outcomes")
    ax.legend(fontsize=8, ncol=5, loc="upper center")
    fig.savefig(FIG / "phaseC_state_outcomes.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---------------- report ----------------
    lines = [
        "# Stage 15 · 阶段 C：当前专家选择对长期导航结果的影响",
        "",
        f"- 共同初始状态：**{n_states}**（100 个选中状态中成功恢复并完成分支）",
        "- 分支设置：当前步使用专家 X，之后固定使用 **TS**；真实闭环至终止",
        "- 状态来自 Val-Seen 任务的 TS 闭环轨迹快照（仅用于离线反事实诊断，"
        "未参与任何训练或调参）",
        "- 状态恢复一致性测试：10/10 动作一致、位置误差 0.0 m",
        "",
        "## C1. 单步专家选择对最终成功率的影响",
        "",
        "| 首步专家 | rollouts | 最终 SR | 平均步数 | 平均 NE (m) | 平均 SPL | 越界 | 碰撞 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in KEYS:
        r = agg.loc[k]
        lines.append(f"| {NAMES[k]} | {int(r.rollouts)} | {r.sr*100:.1f}% | "
                     f"{r.mean_steps:.1f} | {r.mean_final_dist:.1f} | "
                     f"{r.mean_spl:.3f} | {int(r.oob)} | {int(r.collision)} |")
    lines += [
        "",
        "## C2. 状态敏感性",
        "",
        f"- 不同专家导致不同最终结果的状态：**{sens['states_with_different_outcome']} / {n_states} "
        f"= {sens['pct_sensitive']*100:.1f}%**",
        f"- 五专家全部失败：**{sens['all_fail_states']}**",
        f"- 五专家全部成功：**{sens['all_success_states']}**",
        f"- Small 成功而 TS 失败：**{sens['small_success_ts_fail']}**",
        f"- TS 成功而 Small 失败：**{sens['ts_success_small_fail']}**",
        f"- 仅 Small 成功：{sens['small_only']}；仅 TS 成功：{sens['ts_only']}",
        "",
        "## C3. 静态排名 vs 状态最优",
        "",
        f"- 静态 SR 最优专家：**{NAMES[sens['static_best_expert']]}**",
        f"- 在 {sens['sensitive_states']} 个敏感状态上，静态最优获胜 **{sens['pct_static_best_wins_on_sensitive']*100:.1f}%** "
        f"（{sens['static_best_wins_on_sensitive']}/{sens['sensitive_states']}）",
        f"- 若能在敏感状态上做 oracle 选择，最多再提升 **{sens['oracle_choice_gain_on_sensitive']*100:.1f}%** 的成功率",
        "",
        "## C4. 当前步动作分歧与长期结果",
        "",
        f"- 当前步五专家动作存在分歧的状态：**{sens['branch_step_action_divergence']} / {n_states} "
        f"= {sens['pct_branch_step_action_divergence']*100:.1f}%**",
        "",
        "## 产物",
        "",
        "- `branch_100/branch_rollout_results.csv`",
        "- `branch_100/branch_action_traces.json`",
        "- `single_step_expert_effect.csv`",
        "- `state_winner_sets.csv`",
        "- `phaseC_summary.json`",
        "- 图：`figures/phaseC_*.png`",
        "",
    ]
    (HERE / "COUNTERFACTUAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(sens, indent=2))


if __name__ == "__main__":
    main()
