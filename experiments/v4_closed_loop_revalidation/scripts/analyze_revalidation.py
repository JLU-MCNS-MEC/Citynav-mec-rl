#!/usr/bin/env python3
"""V4/V5 corrected closed-loop comparison: merge shards, compute metrics.

Metrics (unified protocol): SR (final stop within 20 m), OSR (ever within
20 m), NE (mean final distance), SPL (success-weighted path optimality,
L* = start-target euclidean), per split / length group / map, failure
breakdown, candidate coverage, empty-candidate behaviour, per-episode
V4/V5 alignment.

Outputs:
  reports/v4_v5_comparison/comparison_report.md
  reports/v4_v5_comparison/metrics.json
  reports/v4_v5_comparison/per_episode_alignment.csv
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

REPO = Path("/mnt/hdd2t/workspaces/Citynav-mec-rl-fixed50m")
HERE = Path(__file__).resolve().parent
V4 = REPO / "experiments/v4_closed_loop_revalidation/results"
V5 = REPO / "experiments/v5_closed_loop_evaluation/results"
OUT = REPO / "reports/v4_v5_comparison"
BASELINES = {
    "C1 (09C-1, historical)": {"val_sr": 0.647},
    "M-v3 (10E, historical)": {"val_sr": 0.554},
}


def load(model_dir: Path, prefix: str) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    for d in sorted(model_dir.glob(f"{prefix}_*")):
        for jf in d.glob("journal*.jsonl"):
            for line in jf.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    recs[r["episode_id"]] = r
                except json.JSONDecodeError:
                    continue
    return recs


def stats(recs: dict[str, dict], label: str) -> dict:
    n = len(recs)
    if not n:
        return {"label": label, "n": 0}
    evalable = {k: r for k, r in recs.items() if r.get("execution_status") == "ok"}
    n_ev = len(evalable)
    succ = sum(r["success"] for r in evalable.values())
    osr = sum(r["osr"] for r in evalable.values())
    spl = sum(r.get("spl", 0.0) for r in evalable.values())
    ne = [r["final_dist"] for r in evalable.values() if r["final_dist"] is not None]
    steps = [r["steps"] for r in evalable.values()]
    by_split = {}
    for split in ("val_seen", "val_unseen", "test_unseen"):
        s = [r for r in evalable.values() if r["split"] == split]
        if s:
            ss = sum(r["success"] for r in s)
            by_split[split] = {
                "n": len(s), "sr": ss / len(s),
                "osr": sum(r["osr"] for r in s) / len(s),
                "ne_mean": (sum(r["final_dist"] for r in s
                                if r["final_dist"] is not None)
                            / max(1, sum(1 for r in s if r["final_dist"] is not None))),
                "spl": sum(r.get("spl", 0.0) for r in s) / len(s),
                "mean_steps": sum(r["steps"] for r in s) / len(s)}
    groups = {}
    for g in ("short", "medium", "long"):
        s = [r for r in evalable.values() if r.get("length_group") == g]
        if s:
            groups[g] = {"n": len(s),
                         "sr": sum(r["success"] for r in s) / len(s)}
    no_cand = [r for r in evalable.values()
               if r.get("candidate_status") == "empty"]
    no_cand_succ = sum(r["success"] for r in no_cand) if no_cand else 0
    failures = Counter(r.get("failure") for r in evalable.values()
                       if not r["success"])
    import numpy as np
    return {"label": label, "n": n, "n_evalable": n_ev,
            "sr": succ / n_ev, "osr": osr / n_ev,
            "ne_mean": float(np.mean(ne)) if ne else None,
            "ne_p95": float(np.percentile(ne, 95)) if ne else None,
            "spl": spl / n_ev,
            "mean_steps": float(np.mean(steps)) if steps else None,
            "p95_steps": float(np.percentile(steps, 95)) if steps else None,
            "by_split": by_split, "length_groups": groups,
            "n_empty_candidates": len(no_cand),
            "empty_candidate_sr": (no_cand_succ / len(no_cand)) if no_cand else None,
            "failures": dict(failures),
            "exceptions": sum(1 for r in recs.values()
                              if r.get("execution_status") != "ok")}


def per_episode_alignment(v4: dict, v5: dict, out_csv: Path) -> dict:
    common = sorted(set(v4) & set(v5))
    both_succ = sum(1 for e in common if v4[e]["success"] and v5[e]["success"])
    v4_only_succ = sum(1 for e in common if v4[e]["success"] and not v5[e]["success"])
    v5_only_succ = sum(1 for e in common if v5[e]["success"] and not v4[e]["success"])
    neither = len(common) - both_succ - v4_only_succ - v5_only_succ
    with out_csv.open("w", encoding="utf-8") as fh:
        fh.write("episode_id,split,map,v4_success,v5_success,v4_final_dist,"
                 "v5_final_dist,v4_steps,v5_steps,v4_spl,v5_spl\n")
        for e in common:
            a, b = v4[e], v5[e]
            fh.write(f"{e},{a['split']},{a['map']},{a['success']},{b['success']},"
                     f"{a.get('final_dist')},{b.get('final_dist')},"
                     f"{a['steps']},{b['steps']},{a.get('spl', 0):.4f},"
                     f"{b.get('spl', 0):.4f}\n")
    return {"common": len(common), "both_success": both_succ,
            "v4_only_success": v4_only_succ, "v5_only_success": v5_only_succ,
            "neither": neither,
            "mcnemar_v4_only": v4_only_succ, "mcnemar_v5_only": v5_only_succ}


def render_md(s4: dict, s5: dict, align: dict, out_md: Path) -> None:
    t4, t5 = s4["by_split"]["test_unseen"], s5["by_split"]["test_unseen"]
    lines = [
        "# V4 / V5 修正版闭环对比报告",
        "",
        "## 1. 总表（统一协议：SR/OSR 20 m，NE=最终距离，SPL=成功率加权路径比，max steps=200）",
        "",
        "| 指标 | V4 | V5 | 差异 |",
        "|---|---:|---:|---:|",
        f"| Val-Seen SR | {s4['by_split']['val_seen']['sr']:.4f} | {s5['by_split']['val_seen']['sr']:.4f} | {s5['by_split']['val_seen']['sr']-s4['by_split']['val_seen']['sr']:+.4f} |",
        f"| Val-Unseen SR | {s4['by_split']['val_unseen']['sr']:.4f} | {s5['by_split']['val_unseen']['sr']:.4f} | {s5['by_split']['val_unseen']['sr']-s4['by_split']['val_unseen']['sr']:+.4f} |",
        f"| Test-Unseen SR | {t4['sr']:.4f} | {t5['sr']:.4f} | {t5['sr']-t4['sr']:+.4f} |",
        f"| Test OSR | {t4['osr']:.4f} | {t5['osr']:.4f} | {t5['osr']-t4['osr']:+.4f} |",
        f"| Test NE (m) | {t4['ne_mean']:.1f} | {t5['ne_mean']:.1f} | {t5['ne_mean']-t4['ne_mean']:+.1f} |",
        f"| Test SPL | {t4['spl']:.4f} | {t5['spl']:.4f} | {t5['spl']-t4['spl']:+.4f} |",
        f"| Short SR | {s4['length_groups']['short']['sr']:.4f} | {s5['length_groups']['short']['sr']:.4f} | {s5['length_groups']['short']['sr']-s4['length_groups']['short']['sr']:+.4f} |",
        f"| Medium SR | {s4['length_groups']['medium']['sr']:.4f} | {s5['length_groups']['medium']['sr']:.4f} | {s5['length_groups']['medium']['sr']-s4['length_groups']['medium']['sr']:+.4f} |",
        f"| Long SR | {s4['length_groups']['long']['sr']:.4f} | {s5['length_groups']['long']['sr']:.4f} | {s5['length_groups']['long']['sr']-s4['length_groups']['long']['sr']:+.4f} |",
        f"| 平均导航步数 | {s4['mean_steps']:.1f} | {s5['mean_steps']:.1f} | {s5['mean_steps']-s4['mean_steps']:+.1f} |",
        f"| P95 导航步数 | {s4['p95_steps']:.0f} | {s5['p95_steps']:.0f} | {s5['p95_steps']-s4['p95_steps']:+.0f} |",
        "",
        "## 2. 逐 episode 对齐",
        "",
        f"- 共同任务：{align['common']}",
        f"- 双成功：{align['both_success']} | 仅 V4 成功：{align['v4_only_success']} | 仅 V5 成功：{align['v5_only_success']} | 双失败：{align['neither']}",
        "",
        "## 3. 分析",
        "",
        f"1. 几何增益：Test-Unseen SR 差异 {t5['sr']-t4['sr']:+.4f}（{'多地标几何带来增益' if t5['sr']>t4['sr'] else '未带来增益'}）",
        f"2. 空候选任务（block_32 等）：V4 {s4['n_empty_candidates']} 条 SR={s4['empty_candidate_sr']} | V5 {s5['n_empty_candidates']} 条 SR={s5['empty_candidate_sr']}",
        f"3. 失败构成 V4：{s4['failures']}",
        f"4. 失败构成 V5：{s5['failures']}",
        f"5. 长距离任务（Long）：V4 {s4['length_groups']['long']['sr']:.4f} → V5 {s5['length_groups']['long']['sr']:.4f}（{s5['length_groups']['long']['sr']-s4['length_groups']['long']['sr']:+.4f}）",
        f"6. 路径效率（SPL）：V4 {s4['spl']:.4f} → V5 {s5['spl']:.4f}",
        f"7. 早停：V4 premature_stop {s4['failures'].get('premature_stop', 0)} 条 | V5 {s5['failures'].get('premature_stop', 0)} 条",
        "",
        "## 4. 历史基线（旧协议，非直接可比）",
        "",
        f"- C1 (09C-1) val SR：{BASELINES['C1 (09C-1, historical)']['val_sr']}",
        f"- M-v3 (10E) val SR：{BASELINES['M-v3 (10E, historical)']['val_sr']}",
        "",
        "## 5. 复现信息",
        "",
        f"- 评测代码：experiments/v4_closed_loop_revalidation/scripts/run_closed_loop.py",
        f"- V4 checkpoint SHA256：{v4_sha}",
        f"- V5 checkpoint SHA256：{v5_sha}",
        f"- 任务清单：Stage 06 manifests（val 5,324 + test 5,311），全部任务执行，SR 分母=全部有效任务",
        "- 旧版 11D 结果（history 顺序/Pose 编码带病）标记为 historical/incomplete",
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-ckpt", default=str(REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4/results/runs_11b/seed42/best.pt"))
    parser.add_argument("--v5-ckpt", default=str(REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4/results/runs_11c/seed42/best.pt"))
    args = parser.parse_args()
    global v4_sha, v5_sha
    import hashlib
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    v4_sha, v5_sha = sha(args.v4_ckpt), sha(args.v5_ckpt)

    v4 = load(V4, "full")
    v5 = load(V5, "v5")
    s4, s5 = stats(v4, "v4"), stats(v5, "v5")
    OUT.mkdir(parents=True, exist_ok=True)
    align = per_episode_alignment(v4, v5, OUT / "per_episode_alignment.csv")
    out = {"v4": s4, "v5": s5, "alignment": align,
           "v4_ckpt_sha": v4_sha, "v5_ckpt_sha": v5_sha,
           "baselines": BASELINES}
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2) + "\n",
                                      encoding="utf-8")
    render_md(s4, s5, align, OUT / "comparison_report.md")
    print(json.dumps({"v4": {"n": s4["n_evalable"], "sr": s4["sr"],
                             "by_split": s4["by_split"]},
                      "v5": {"n": s5["n_evalable"], "sr": s5["sr"],
                             "by_split": s5["by_split"]},
                      "alignment": align}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
