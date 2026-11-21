#!/usr/bin/env python3
"""Stage 15 Phase A: task-level complementarity of the five MEC experts.

Uses ONLY the existing closed-loop CSVs (no model inference). Aligns strictly
by episode_id, then computes pairwise/portfolio complementarity, McNemar tests,
paired bootstrap CIs, task-group breakdowns and figures.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
EXP14 = ROOT.parents[1] / "14_mec_expert_retraining"
OUT = ROOT
FIG = ROOT.parent / "figures"
FIG.mkdir(exist_ok=True)

MODELS = ["Small", "Medium", "TS", "TM", "Large"]
FLOPS = {"Small": 5, "Medium": 15, "TS": 40, "TM": 100, "Large": 200}
PATHS = {
    "Small": EXP14 / "results/Small/closed_loop/val_seen_episode_results.csv",
    "Medium": EXP14 / "results/Medium/closed_loop/val_seen_episode_results.csv",
    "TS": EXP14 / "stability_revision/eval/ts/closed_loop/val_seen_episode_results.csv",
    "TM": EXP14 / "stability_revision/eval/tm/closed_loop/val_seen_episode_results.csv",
    "Large": EXP14 / "stability_revision/eval/large/closed_loop/val_seen_episode_results.csv",
}
COLORS = {"Small": "#4c78a8", "Medium": "#f58518", "TS": "#54a24b",
          "TM": "#b279a2", "Large": "#e45756"}
RNG = np.random.default_rng(42)
N_BOOT = 10000


def load() -> dict[str, pd.DataFrame]:
    out = {}
    for m, p in PATHS.items():
        df = pd.read_csv(p, dtype={"episode_id": str})
        df["success"] = df["success"].astype(str).str.lower().isin(["true", "1"])
        for c in ("steps", "spl", "ne_m", "path_len_m", "l_star_m"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        out[m] = df.set_index("episode_id")
    return out


def main() -> None:
    dfs = load()
    ids = sorted(set.intersection(*[set(d.index) for d in dfs.values()]))
    assert len(ids) == 2498, f"aligned ids = {len(ids)}"
    n = len(ids)

    # ---------------- A1 success matrix ----------------
    mat = pd.DataFrame(index=ids)
    mat["map"] = dfs["TS"].loc[ids, "map"]
    mat["length_group"] = dfs["TS"].loc[ids, "length_group"]
    mat["cohort"] = dfs["TS"].loc[ids, "cohort"]
    mat["path_len_m"] = dfs["TS"].loc[ids, "path_len_m"]
    for m in MODELS:
        d = dfs[m].loc[ids]
        mat[f"{m}_success"] = d["success"].astype(int)
        mat[f"{m}_steps"] = d["steps"]
        mat[f"{m}_spl"] = d["spl"]
        mat[f"{m}_ne_m"] = d["ne_m"]
        mat[f"{m}_failure"] = d["failure"]
        mat[f"{m}_stop_reason"] = d["stop_reason"]
    mat.to_csv(OUT / "expert_task_success_matrix.csv")

    S = {m: mat[f"{m}_success"].astype(bool).values for m in MODELS}

    # ---------------- A2 pairwise ----------------
    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        A, B = S[a], S[b]
        both = int((A & B).sum()); a_only = int((A & ~B).sum())
        b_only = int((~A & B).sum()); neither = int((~A & ~B).sum())
        union = both + a_only + b_only
        # McNemar exact (binomial) on discordant pairs
        disc = a_only + b_only
        p = float(binomtest(min(a_only, b_only), disc, 0.5).pvalue) if disc else 1.0
        # paired bootstrap CI for SR difference
        diffs = np.empty(N_BOOT)
        idx = RNG.integers(0, n, size=(N_BOOT, n))
        av = A[idx].mean(axis=1); bv = B[idx].mean(axis=1)
        diffs = av - bv
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({
            "model_a": a, "model_b": b,
            "both_success": both, "a_only": a_only, "b_only": b_only,
            "neither": neither, "intersection": both, "union": union,
            "jaccard": both / union if union else 0.0,
            "sr_a": A.mean(), "sr_b": B.mean(),
            "sr_diff_a_minus_b": A.mean() - B.mean(),
            "mcnemar_p": p,
            "sr_diff_ci_low": lo, "sr_diff_ci_high": hi,
        })
    pair = pd.DataFrame(rows).sort_values("jaccard", ascending=False)
    pair.to_csv(OUT / "pairwise_complementarity.csv", index=False)

    # TS-failure rescue counts
    ts_fail = ids_mask = ~S["TS"]
    rescue = {}
    for m in MODELS:
        if m == "TS":
            continue
        rescue[m] = int((S[m] & ts_fail).sum())
    rescue_df = pd.DataFrame([{"model": k, "success_on_TS_failures": v,
                               "pct_of_TS_failures": v / int(ts_fail.sum())}
                              for k, v in rescue.items()])
    rescue_df.to_csv(OUT / "ts_failure_rescue.csv", index=False)

    # ---------------- A3 portfolio ----------------
    # unique success (only this expert)
    unique = []
    for m in MODELS:
        others = np.zeros(n, bool)
        for o in MODELS:
            if o != m:
                others |= S[o]
        uniq = S[m] & ~others
        unique.append({"model": m, "unique_success": int(uniq.sum()),
                       "sr": float(S[m].mean()),
                       "unique_pct_of_all": float(uniq.sum() / n)})
    unique_df = pd.DataFrame(unique)
    unique_df.to_csv(OUT / "expert_unique_success.csv", index=False)

    # success-count distribution
    cnt = np.stack([S[m] for m in MODELS], axis=1).sum(axis=1)
    dist = pd.Series(cnt).value_counts().sort_index()
    dist_df = pd.DataFrame({"n_experts_success": dist.index,
                            "n_tasks": dist.values,
                            "pct": dist.values / n})
    dist_df.to_csv(OUT / "success_count_distribution.csv", index=False)

    # portfolio coverage for all subsets
    port = []
    for r in range(1, 6):
        for combo in itertools.combinations(MODELS, r):
            cov = np.zeros(n, bool)
            for m in combo:
                cov |= S[m]
            port.append({"subset_size": r, "combo": "+".join(combo),
                         "coverage": float(cov.mean()),
                         "success_tasks": int(cov.sum())})
    port_df = pd.DataFrame(port).sort_values("coverage", ascending=False)
    port_df.to_csv(OUT / "expert_portfolio_coverage.csv", index=False)

    # best coverage by subset size + oracle union
    best_by_size = port_df.sort_values("coverage", ascending=False).groupby(
        "subset_size", as_index=False).first()

    # ---------------- A4 task groups ----------------
    group_rows = []
    for dim in ("length_group", "map", "cohort"):
        for g, sub in mat.groupby(dim, dropna=False):
            for m in MODELS:
                col = f"{m}_success"
                group_rows.append({
                    "dimension": dim, "group": str(g), "n_tasks": len(sub),
                    "model": m, "sr": float(sub[col].mean()),
                    "success_tasks": int(sub[col].sum()),
                    "mean_steps": float(sub[f"{m}_steps"].mean()),
                })
    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(OUT / "task_group_analysis.csv", index=False)

    # failure-mode complementarity
    fail_rows = []
    for m in MODELS:
        vc = mat[f"{m}_failure"].fillna("none").value_counts()
        for k, v in vc.items():
            fail_rows.append({"model": m, "failure": str(k), "count": int(v)})
    fail_df = pd.DataFrame(fail_rows)
    fail_df.to_csv(OUT / "failure_mode_breakdown.csv", index=False)

    # ---------------- figures ----------------
    # (1) UpSet-style plot of success-set patterns
    masks = np.zeros(n, dtype=int)
    for i, m in enumerate(MODELS):
        masks |= S[m].astype(int) << i
    combos = pd.Series(masks).value_counts().sort_values(ascending=False)
    combos = combos[combos.index > 0].head(16)
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.4], hspace=0.06)
    ax_bar = fig.add_subplot(gs[0]); ax_dot = fig.add_subplot(gs[1])
    xs = np.arange(len(combos))
    ax_bar.bar(xs, combos.values, color="#4c78a8")
    for x, v in zip(xs, combos.values):
        ax_bar.text(x, v, str(int(v)), ha="center", va="bottom", fontsize=8)
    ax_bar.set_ylabel("tasks"); ax_bar.set_xticks([])
    ax_bar.set_title("Success-set patterns (UpSet-style; by episode_id)")
    ax_dot.set_xlim(-0.6, len(combos) - 0.4); ax_dot.set_ylim(-0.6, len(MODELS) - 0.4)
    ax_dot.set_yticks(range(len(MODELS)))
    ax_dot.set_yticklabels(MODELS[::-1])
    for x, mask in zip(xs, combos.index):
        present = [i for i in range(len(MODELS)) if mask >> i & 1]
        for i in range(len(MODELS)):
            y = len(MODELS) - 1 - i
            ax_dot.plot(x, y, "o", ms=7,
                        color="#222" if i in present else "#ddd")
        if present:
            ax_dot.plot([x, x], [len(MODELS) - 1 - max(present),
                                 len(MODELS) - 1 - min(present)], "-", color="#222", lw=2)
    ax_dot.set_xticks([]); ax_dot.set_xlabel("pattern (sorted by task count)")
    fig.savefig(FIG / "phaseA_upset.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # (2) pairwise Jaccard heatmap
    hm = np.ones((5, 5))
    for r in pair.itertuples():
        i, j = MODELS.index(r.model_a), MODELS.index(r.model_b)
        hm[i, j] = hm[j, i] = r.jaccard
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(hm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(MODELS); ax.set_yticklabels(MODELS)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{hm[i, j]:.3f}", ha="center", va="center",
                    color="white" if hm[i, j] < 0.6 else "black", fontsize=9)
    fig.colorbar(im, label="success-set Jaccard")
    ax.set_title("Pairwise success-set Jaccard")
    fig.savefig(FIG / "phaseA_jaccard_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # (3) subset-size coverage curve
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.plot(best_by_size["subset_size"], best_by_size["coverage"], "o-", label="best subset")
    mean_cov = port_df.groupby("subset_size")["coverage"].mean()
    ax.plot(mean_cov.index, mean_cov.values, "s--", label="mean subset")
    for m in MODELS:
        ax.axhline(S[m].mean(), color=COLORS[m], lw=0.8, alpha=0.5)
        ax.text(1.02, S[m].mean(), m, color=COLORS[m], fontsize=8, va="center")
    ax.set_xlabel("number of experts in fixed portfolio")
    ax.set_ylabel("task coverage (union SR)")
    ax.set_title("Portfolio coverage vs subset size")
    ax.grid(alpha=.3); ax.legend()
    fig.savefig(FIG / "phaseA_coverage_curve.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # (4) FLOPs vs SR
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for m in MODELS:
        ax.scatter(FLOPS[m], S[m].mean(), s=80, color=COLORS[m], label=m)
        ax.annotate(m, (FLOPS[m], S[m].mean()), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("FLOPs (relative)")
    ax.set_ylabel("independent SR"); ax.grid(alpha=.3)
    ax.set_title("FLOPs vs independent SR")
    fig.savefig(FIG / "phaseA_flops_vs_sr.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # (5) all-subset coverage bars (top 15)
    top = port_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(range(len(top)), top["coverage"], color="#4c78a8")
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top["combo"], fontsize=8)
    ax.set_xlabel("coverage (union SR)"); ax.set_xlim(0, 1)
    for i, v in enumerate(top["coverage"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    ax.set_title("Top expert portfolios by task coverage")
    fig.savefig(FIG / "phaseA_portfolio_coverage.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---------------- report ----------------
    ts_fail_n = int(ts_fail.sum())
    all_success = int((cnt == 5).sum()); all_fail = int((cnt == 0).sum())
    oracle_union = float(np.any(np.stack([S[m] for m in MODELS], 1), 1).mean())
    ts_small_row = pair[((pair.model_a == "TS") & (pair.model_b == "Small")) |
                        ((pair.model_a == "Small") & (pair.model_b == "TS"))]
    ts_small = ts_small_row.iloc[0]
    ts_minus_small = (ts_small.sr_diff_a_minus_b
                      if ts_small.model_a == "TS" else -ts_small.sr_diff_a_minus_b)
    lines = [
        "# Stage 15 · 阶段 A：任务级专家互补性分析",
        "",
        f"- 对齐任务数：{n}（严格按 episode_id）",
        f"- manifest sha：2da3781677e1…（五模型一致）",
        f"- step cap 200，success radius 20 m，exceptions 0",
        "",
        "## A1. 独立成功率",
        "",
        "| 模型 | FLOPs | SR | 成功数 |",
        "|---|---:|---:|---:|",
    ]
    for m in MODELS:
        lines.append(f"| {m} | {FLOPS[m]}G | {S[m].mean()*100:.2f}% | {int(S[m].sum())} |")
    lines += [
        "",
        "## A2. TS 失败任务上的救援能力（关键结果）",
        "",
        f"TS 失败任务数：**{ts_fail_n}**",
        "",
        "| 专家 | 在 TS 失败任务上的成功数 | 占 TS 失败任务比例 |",
        "|---|---:|---:|",
    ]
    for r in rescue_df.itertuples():
        lines.append(f"| {r.model} | {r.success_on_TS_failures} | "
                     f"{r.pct_of_TS_failures*100:.2f}% |")
    lines += [
        "",
        "## A2b. 两两互补性（节选）",
        "",
        "| 组合 | 交集 | 并集 | Jaccard | SR 差 (A−B) | McNemar p | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in pair.head(10).itertuples():
        lines.append(f"| {r.model_a} vs {r.model_b} | {r.intersection} | {r.union} | "
                     f"{r.jaccard:.3f} | {r.sr_diff_a_minus_b:+.4f} | {r.mcnemar_p:.2e} | "
                     f"[{r.sr_diff_ci_low:+.4f}, {r.sr_diff_ci_high:+.4f}] |")
    lines += [
        "",
        f"TS vs Small：交集 {ts_small.intersection}，并集 {ts_small.union}，"
        f"Jaccard {ts_small.jaccard:.3f}，SR 差 (TS−Small) {ts_minus_small:+.4f}"
        f"（McNemar p={ts_small.mcnemar_p:.2e}）。",
        "",
        "## A3. 整体互补性",
        "",
        f"- 五模型全部成功：**{all_success}** 条",
        f"- 五模型全部失败：**{all_fail}** 条",
        f"- 五模型成功集合的并集覆盖率（事后 oracle）：**{oracle_union*100:.2f}%**",
        "",
        "| 成功专家数 | 任务数 | 占比 |",
        "|---:|---:|---:|",
    ]
    for r in dist_df.itertuples():
        lines.append(f"| {r.n_experts_success} | {r.n_tasks} | {r.pct*100:.2f}% |")
    lines += [
        "",
        "| 模型 | 独有成功任务数 | 占全部任务 |",
        "|---|---:|---:|",
    ]
    for r in unique_df.itertuples():
        lines.append(f"| {r.model} | {r.unique_success} | {r.unique_pct_of_all*100:.2f}% |")
    lines += [
        "",
        "**重要说明**：上述并集覆盖率是“离线已知每条任务结果后固定选择专家”的事后参考值，"
        "不是可部署的动态专家关联策略 SR，也不是逐步切换专家的严格上界。",
        "",
        "## A4. 任务分组（length_group 为路径长度分组，不等同于难度）",
        "",
    ]
    lg = group_df[group_df.dimension == "length_group"]
    if not lg.empty:
        pivot = lg.pivot(index="group", columns="model", values="sr")
        lines.append("| length_group | n | " + " | ".join(MODELS) + " |")
        lines.append("|---|---:|" + "---:|" * len(MODELS))
        for g, row in pivot.iterrows():
            nsub = int(lg[(lg.group == g) & (lg.model == MODELS[0])]["n_tasks"].iloc[0])
            lines.append(f"| {g} | {nsub} | " + " | ".join(
                f"{row.get(m, np.nan)*100:.2f}%" for m in MODELS) + " |")
    lines += ["", "## 产物", "",
              "- `expert_task_success_matrix.csv`", "- `pairwise_complementarity.csv`",
              "- `expert_unique_success.csv`", "- `expert_portfolio_coverage.csv`",
              "- `task_group_analysis.csv`", "- `failure_mode_breakdown.csv`",
              "- `ts_failure_rescue.csv`", "- `success_count_distribution.csv`",
              "- 图：`figures/phaseA_*.png`", ""]
    (OUT / "TASK_LEVEL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "n_tasks": n,
        "independent_sr": {m: float(S[m].mean()) for m in MODELS},
        "ts_failures": ts_fail_n,
        "rescue_on_ts_failures": rescue,
        "all_success": all_success, "all_fail": all_fail,
        "oracle_union_sr": oracle_union,
        "best_subset_by_size": best_by_size.to_dict("records"),
        "unique_success": dict(zip(unique_df.model, unique_df.unique_success)),
    }
    (OUT / "phaseA_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
