#!/usr/bin/env python3
"""10B closed-loop comparison: C1 vs PQ vs CA (paired statistics).

Reads journals from:
* 09C-1:  experiments/09_.../09C_continuous_localization/results/09c_c1/journal.jsonl
* 10B:    results/10b_pq/journal.jsonl, results/10b_ca/journal.jsonl
          (and results/10b_c1/journal.jsonl if the harness-validation run
          produced one)

Outputs: results/10b_comparison.json, results/10b_comparison_tables.csv,
figures/10b_sr_bars.png, figures/10b_paired.png, figures/10b_trajectories.png.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"
J09C_C1 = (REPO / "experiments/09_online_visual_goal_matching"
           "/09C_continuous_localization/results/09c_c1/journal.jsonl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {"09c_c1": "C1 (09C-1)", "10b_c1": "C1 (10B harness)",
          "10b_pq": "PQ", "10b_ca": "CA"}
COLORS = {"09c_c1": "#2c7fb8", "10b_c1": "#6baed6", "10b_pq": "#d95f0e",
          "10b_ca": "#31a354"}


def load_journal(path: Path) -> dict[str, dict]:
    out = {}
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
    """Two-sided exact McNemar p-value on paired boolean vectors."""
    b01 = int(np.sum(~a & b))
    b10 = int(np.sum(a & ~b))
    n_disc = b01 + b10
    if n_disc == 0:
        return 1.0
    from scipy.stats import binom
    p = 2 * min(1.0, float(binom.cdf(min(b01, b10), n_disc, 0.5))
                + (0.5 * binom.pmf(min(b01, b10), n_disc, 0.5)
                   if b01 != b10 else 0.0))
    return float(p)


def stats(journal: dict) -> dict:
    recs = list(journal.values())
    n = len(recs)
    succ = np.array([r["success"] for r in recs], bool)
    osr = np.array([r["osr"] for r in recs], bool)
    oob = np.array([r["out_of_bounds"] for r in recs], bool)
    early = np.array([r["stopped"] and (r["stop_dist"] or 1e9) > 20.0
                      for r in recs], bool)
    steps = np.array([r["steps"] for r in recs], float)
    fdist = np.array([r["final_dist"] for r in recs], float)
    acts = np.zeros(4)
    for r in recs:
        for i, k in enumerate(["Stop", "Forward", "TurnLeft", "TurnRight"]):
            acts[i] += r["action_counts"].get(k, 0)
    lo, hi = wilson(int(succ.sum()), n)
    out = {"n": n, "sr": float(succ.mean()), "sr_lo": lo, "sr_hi": hi,
           "osr": float(osr.mean()), "oob": float(oob.mean()),
           "early_stop": float(early.mean()),
           "avg_steps": float(steps.mean()), "avg_final_dist": float(fdist.mean()),
           "action_frac": (acts / max(acts.sum(), 1)).tolist()}
    for split in ("val_seen", "val_unseen"):
        m = np.array([r["split"] == split for r in recs])
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
    journals = {}
    if J09C_C1.exists():
        journals["09c_c1"] = load_journal(J09C_C1)
    for key in ("10b_c1", "10b_pq", "10b_ca"):
        p = RES / key / "journal.jsonl"
        if p.exists():
            journals[key] = load_journal(p)
    print(f"journals: { {k: len(v) for k, v in journals.items()} }", flush=True)

    summary = {k: stats(v) for k, v in journals.items()}
    (RES / "10b_comparison.json").write_text(json.dumps(
        {"summary": summary, "labels": LABELS}, indent=2) + "\n", encoding="utf-8")

    # paired comparisons vs 09c_c1
    rows = []
    ref = journals.get("09c_c1")
    for key in ("10b_c1", "10b_pq", "10b_ca"):
        if key not in journals or ref is None:
            continue
        ids = sorted(set(ref) & set(journals[key]))
        a = np.array([ref[i]["success"] for i in ids], bool)
        b = np.array([journals[key][i]["success"] for i in ids], bool)
        diff = b.mean() - a.mean()
        lo, hi = wilson(int(b.sum()), len(b))
        p = mcnemar(a, b)
        rows.append({"pair": f"{key}_vs_09c_c1", "n_paired": len(ids),
                     "sr_ref": float(a.mean()), "sr_other": float(b.mean()),
                     "sr_other_lo": lo, "sr_other_hi": hi,
                     "delta_pp": round(float(diff) * 100, 2),
                     "mcnemar_p": f"{p:.3g}",
                     "wins": int((b & ~a).sum()), "losses": int((a & ~b).sum())})
        print(f"{key} vs 09c_c1: n={len(ids)} ref={a.mean():.3f} "
              f"other={b.mean():.3f} delta={diff * 100:+.2f}pp p={p:.3g}",
              flush=True)
    with (RES / "10b_comparison_tables.csv").open("w", newline="",
                                                  encoding="utf-8") as fh:
        import csv
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- figures -----------------------------------------------------------
    keys = list(journals)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, k in enumerate(keys):
        s = summary[k]
        ax.bar(i, s["sr"] * 100, yerr=[[(s["sr"] - s["sr_lo"]) * 100],
                                       [(s["sr_hi"] - s["sr"]) * 100]],
               capsize=5, color=COLORS[k], alpha=0.9)
        ax.text(i, s["sr"] * 100 + 1.5, f"{s['sr'] * 100:.1f}%", ha="center")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([LABELS[k] for k in keys])
    ax.set_ylabel("SR (%)  [Val-Seen + Val-Unseen]")
    ax.set_ylim(0, 80)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("10B: closed-loop success rate (5,324 tasks)")
    fig.tight_layout()
    fig.savefig(FIG / "10b_sr_bars.png", dpi=150)
    plt.close(fig)

    # per-split and per-length SR
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
        ax.legend(fontsize=8)
    fig.suptitle("10B: SR breakdown")
    fig.tight_layout()
    fig.savefig(FIG / "10b_breakdown.png", dpi=150)
    plt.close(fig)

    # paired scatter (success rate per pair)
    if ref is not None and len(rows) > 1:
        fig, axes = plt.subplots(1, len(rows), figsize=(4.5 * len(rows), 4))
        if len(rows) == 1:
            axes = [axes]
        for ax, row, key in zip(axes, rows, [k for k in
                                             ("10b_c1", "10b_pq", "10b_ca")
                                             if k in journals and ref is not None]):
            ids = sorted(set(ref) & set(journals[key]))
            a = np.array([ref[i]["success"] for i in ids], bool)
            b = np.array([journals[key][i]["success"] for i in ids], bool)
            vals, cnts = np.unique(np.stack([a, b], axis=1).astype(int)
                                   @ np.array([1, 2]), return_counts=True)
            m = {"00": 0, "01": 0, "10": 0, "11": 0}
            for v, c in zip(vals, cnts):
                m[["00", "01", "10", "11"][int(v)]] = int(c)
            ax.bar([0, 1, 2, 3], [m["00"], m["01"], m["10"], m["11"]],
                   color=["#aaa", COLORS[key], "#666", "#222"])
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(["both fail", "only other", "only C1", "both win"],
                               fontsize=7, rotation=20)
            ax.set_title(f"{LABELS[key]} vs C1\nMcNemar p={row['mcnemar_p']}",
                         fontsize=9)
        fig.suptitle("10B: paired outcomes vs C1 (09C-1)")
        fig.tight_layout()
        fig.savefig(FIG / "10b_paired.png", dpi=150)
        plt.close(fig)

    # trajectory examples: pick cases where outcomes differ
    if "10b_pq" in journals and "10b_ca" in journals and ref is not None:
        ids = sorted(set(ref) & set(journals["10b_pq"]) & set(journals["10b_ca"]))
        diff = [i for i in ids
                if not (ref[i]["success"] == journals["10b_pq"][i]["success"]
                        == journals["10b_ca"][i]["success"])]
        rng = np.random.default_rng(42)
        picks = list(rng.choice(diff, size=min(3, len(diff)), replace=False)) \
            if diff else []
        if picks:
            fig, axes = plt.subplots(3, len(picks), figsize=(4.5 * len(picks), 11))
            if len(picks) == 1:
                axes = axes[:, None]
            for j, eid in enumerate(picks):
                for i, k in enumerate(("09c_c1", "10b_pq", "10b_ca")):
                    ax = axes[i, j]
                    rec = journals[k][eid]
                    xs = [s["x"] for s in rec["trajectory"]]
                    ys = [s["y"] for s in rec["trajectory"]]
                    ax.plot(xs, ys, ".-", color=COLORS[k], ms=3, lw=1)
                    ax.plot(rec["target"][0], rec["target"][1], "g*", ms=14)
                    ax.set_title(f"{LABELS[k]} | {'SUCCESS' if rec['success'] else 'fail'} "
                                 f"| {rec['steps']} steps", fontsize=8)
                    ax.set_axis_off()
                axes[0, j].set_title(f"{eid}\n{split_title(ref[eid])}", fontsize=8)
            fig.suptitle("10B: trajectory examples (divergent outcomes)")
            fig.tight_layout()
            fig.savefig(FIG / "10b_trajectories.png", dpi=150)
            plt.close(fig)

    print("10B analysis written", flush=True)
    return 0


def split_title(rec: dict) -> str:
    return f"{rec['split']} | {rec['length_group']} | dist {rec['final_dist']:.0f} m"


if __name__ == "__main__":
    raise SystemExit(main())
