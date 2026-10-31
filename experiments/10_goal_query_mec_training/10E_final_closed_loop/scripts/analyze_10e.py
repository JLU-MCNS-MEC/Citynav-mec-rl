#!/usr/bin/env python3
"""10E analysis: 5 controllers on the same 5,324 tasks (paired).

Controllers:
* stage06   - old Transformer-M v2 (Stage 06 journal)
* top1      - Stage 08 Top-1 geometric baseline
* dyn09b    - Stage 09B dynamic geometric baseline
* c1_09c    - Stage 09C-1 continuous localisation (formal baseline)
* m10e      - Stage 10D Transformer-M v3 (this stage)

Outputs: results/10e_comparison.json, 10e_comparison_tables.csv,
figures/10e_sr_bars.png, 10e_breakdown.png, 10e_trajectories.png.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
assert (REPO / "experiments").is_dir(), f"REPO 解析错误: {REPO}"
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SOURCES = {
    "stage06": REPO / "experiments/06_closed_loop_success_rate/results/seed42/journal.jsonl",
    "top1": REPO / "experiments/08_belief_only_geometric_navigation/results/top1/journal.jsonl",
    "dyn09b": REPO / "experiments/09_online_visual_goal_matching/results/09b_dynamic/journal.jsonl",
    "c1_09c": REPO / "experiments/09_online_visual_goal_matching/09C_continuous_localization/results/09c_c1/journal.jsonl",
    "m10e": RES / "10e_loop" / "journal.jsonl",
}
LABELS = {"stage06": "Stage06 M-v2", "top1": "Stage08 Top-1",
          "dyn09b": "Stage09B dyn", "c1_09c": "C1 09C-1", "m10e": "Stage10 M-v3"}
COLORS = {"stage06": "#8c564b", "top1": "#7f7f7f", "dyn09b": "#bcbd22",
          "c1_09c": "#2c7fb8", "m10e": "#d62728"}


def load_journal(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[r["episode_id"]] = r
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    b01 = int(np.sum(~a & b))
    b10 = int(np.sum(a & ~b))
    n_disc = b01 + b10
    if n_disc == 0:
        return 1.0
    from scipy.stats import binom
    return float(2 * min(1.0, binom.cdf(min(b01, b10), n_disc, 0.5)
                         + (0.5 * binom.pmf(min(b01, b10), n_disc, 0.5)
                            if b01 != b10 else 0.0)))


def stats(journal: dict) -> dict:
    recs = list(journal.values())
    n = len(recs)
    succ = np.array([r["success"] for r in recs], bool)
    osr = np.array([r["osr"] for r in recs], bool)
    oob = np.array([r["out_of_bounds"] for r in recs], bool)
    early = np.array([r.get("stopped", False) and (r.get("stop_dist") or 1e9) > 20.0
                      for r in recs], bool)
    steps = np.array([r["steps"] for r in recs], float)
    fdist = np.array([r["final_dist"] for r in recs], float)
    lo, hi = wilson(int(succ.sum()), n)
    out = {"n": n, "sr": float(succ.mean()), "sr_lo": lo, "sr_hi": hi,
           "osr": float(osr.mean()), "oob": float(oob.mean()),
           "early_stop": float(early.mean()),
           "avg_steps": float(steps.mean()),
           "avg_final_dist": float(fdist.mean())}
    for split in ("val_seen", "val_unseen"):
        m = np.array([r["split"] == split for r in recs])
        if m.sum():
            lo, hi = wilson(int(succ[m].sum()), int(m.sum()))
            out[f"{split}_sr"] = float(succ[m].mean())
            out[f"{split}_sr_lo"], out[f"{split}_sr_hi"] = lo, hi
    for lg in ("short", "medium", "long"):
        m = np.array([r["length_group"] == lg for r in recs])
        if m.sum():
            out[f"{lg}_sr"] = float(succ[m].mean())
    return out


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    journals = {k: load_journal(p) for k, p in SOURCES.items()}
    # Stage 06 journal was recovered from the migration archive
    # (results_06/results/seed42/journal.jsonl, 5,324 lines) and copied to
    # results/stage06_journal.jsonl for the paired comparison
    s6_j = RES / "stage06_journal.jsonl"
    if not journals.get("stage06") and s6_j.exists():
        journals["stage06"] = load_journal(s6_j)
    s6_path = (REPO / "experiments/06_closed_loop_success_rate/results/seed42"
               "/summary.json")
    if not journals.get("stage06") and s6_path.exists():
        pass
    print("journals:", {k: len(v) for k, v in journals.items()}, flush=True)
    summary = {k: stats(v) for k, v in journals.items() if v}
    if not summary.get("stage06") and s6_path.exists():
        sc = json.loads(s6_path.read_text(encoding="utf-8"))["scopes"]
        all_ = sc["all"]
        s6 = {"n": int(all_["n"]), "sr": float(all_["sr"]),
              "sr_lo": float(all_["sr_ci95"][0]), "sr_hi": float(all_["sr_ci95"][1]),
              "osr": float(all_["osr"]),
              "oob": float(all_["out_of_bounds_rate"]),
              "early_stop": float(all_["premature_stop_rate"]),
              "avg_steps": float(all_["steps_mean"]),
              "avg_final_dist": float(all_["ne_mean"]),
              "aggregated_only": True}
        for split in ("val_seen", "val_unseen"):
            if split in sc:
                s6[f"{split}_sr"] = float(sc[split]["sr"])
        for lg in ("short", "medium", "long"):
            if f"all|{lg}" in sc:
                s6[f"{lg}_sr"] = float(sc[f"all|{lg}"]["sr"])
    (RES / "10e_comparison.json").write_text(json.dumps(
        {"summary": summary, "labels": LABELS}, indent=2) + "\n", encoding="utf-8")

    # paired vs the new M
    rows = []
    ref = journals.get("m10e", {})
    for key in ("top1", "dyn09b", "c1_09c", "stage06"):
        if key not in journals or not ref:
            continue
        ids = sorted(set(ref) & set(journals[key]))
        if not ids:
            continue
        a = np.array([ref[i]["success"] for i in ids], bool)
        b = np.array([journals[key][i]["success"] for i in ids], bool)
        diff = b.mean() - a.mean()
        lo, hi = wilson(int(b.sum()), len(b))
        p = mcnemar(a, b)
        rows.append({"pair": f"m10e_vs_{key}", "n_paired": len(ids),
                     "sr_m10e": float(a.mean()), "sr_other": float(b.mean()),
                     "sr_other_lo": lo, "sr_other_hi": hi,
                     "delta_pp": round(float(diff) * 100, 2),
                     "mcnemar_p": f"{p:.3g}",
                     "wins": int((b & ~a).sum()), "losses": int((a & ~b).sum())})
        print(f"m10e vs {key}: n={len(ids)} m10e={a.mean():.3f} "
              f"other={b.mean():.3f} delta={diff * 100:+.2f}pp p={p:.3g}",
              flush=True)
    if rows:
        import csv
        with (RES / "10e_comparison_tables.csv").open("w", newline="",
                                                      encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # ---- figures -----------------------------------------------------------
    keys = [k for k in journals if journals[k]]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, k in enumerate(keys):
        s = summary[k]
        ax.bar(i, s["sr"] * 100,
               yerr=[[(s["sr"] - s["sr_lo"]) * 100], [(s["sr_hi"] - s["sr"]) * 100]],
               capsize=5, color=COLORS[k], alpha=0.9)
        ax.text(i, s["sr"] * 100 + 1.2, f"{s['sr'] * 100:.1f}%", ha="center")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([LABELS[k] for k in keys], fontsize=9)
    ax.set_ylabel("SR (%)  [5,324 tasks]")
    ax.set_ylim(0, 90)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("10E: closed-loop SR across controllers")
    fig.tight_layout()
    fig.savefig(FIG / "10e_sr_bars.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (title, groups) in zip(axes, (
            ("SR by split", ("val_seen", "val_unseen")),
            ("SR by length", ("short", "medium", "long")))):
        x = np.arange(len(groups))
        width = 0.8 / len(keys)
        for i, k in enumerate(keys):
            vals = [summary[k].get(f"{g}_sr", 0) * 100 for g in groups]
            ax.bar(x + (i - len(keys) / 2 + 0.5) * width, vals, width,
                   color=COLORS[k], label=LABELS[k], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)
    fig.suptitle("10E: SR breakdown")
    fig.tight_layout()
    fig.savefig(FIG / "10e_breakdown.png", dpi=150)
    plt.close(fig)

    # trajectories: 3 success + 3 fail from the new M
    if "m10e" in journals:
        recs = list(journals["m10e"].values())
        succ = [r for r in recs if r["success"]]
        fail = [r for r in recs if not r["success"]]
        rng = np.random.default_rng(42)
        ps = list(rng.choice(succ, size=min(3, len(succ)), replace=False)) if succ else []
        pf = list(rng.choice(fail, size=min(3, len(fail)), replace=False)) if fail else []
        ep_dir = RES / "10e_loop" / "episodes"
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
        for j, (rec, ax) in enumerate(zip(ps + pf, axes.flat)):
            p = ep_dir / f"{rec['episode_id'].replace(':', '_')}.json"
            traj = json.loads(p.read_text(encoding="utf-8"))["trajectory"] \
                if p.exists() else []
            xs = [s["x"] for s in traj]
            ys = [s["y"] for s in traj]
            ax.plot(xs, ys, ".-", color=COLORS["m10e"], ms=3, lw=1)
            ax.plot(rec["target"][0], rec["target"][1], "g*", ms=14)
            tag = "SUCCESS" if rec["success"] else "fail"
            ax.set_title(f"{rec['episode_id']} | {tag} | {rec['steps']} steps | "
                         f"final {rec['final_dist']:.0f} m", fontsize=8)
            ax.set_axis_off()
        fig.suptitle("10E: Transformer-M v3 trajectories (3 success + 3 fail)")
        fig.tight_layout()
        fig.savefig(FIG / "10e_trajectories.png", dpi=150)
        plt.close(fig)

    print("10E analysis written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
