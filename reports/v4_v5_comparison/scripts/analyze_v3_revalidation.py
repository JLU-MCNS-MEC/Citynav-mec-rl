#!/usr/bin/env python3
"""V4 / V3 corrected-protocol comparison (val only) — merge shards, metrics.

V3 here is the Stage-10D M-v3 model RE-EVALUATED with the corrected
protocol (corrected-history-pose-emptycand) on the same 5,324-task val
manifest as the official V4/V5 runs, so V4-vs-V3 numbers are strictly
comparable. The historical 10E M-v3 numbers are reported only as an
old-protocol baseline, never mixed into these metrics.

Outputs:
  reports/v4_v5_comparison/v4_v3_alignment.csv
  reports/v4_v5_comparison/v3_revalidation_metrics.json
  reports/v4_v5_comparison/v3_comparison_report.md
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
assert (REPO / "experiments").is_dir(), f"REPO 解析错误: {REPO}"
V4 = REPO / "experiments/v4_closed_loop_revalidation/results"
V3 = REPO / "experiments/v3_closed_loop_revalidation/results"
OUT = REPO / "reports/v4_v5_comparison"
HISTORICAL_BASELINE = {"M-v3 (10E, old protocol)": {"val_sr": 0.554}}


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
    evalable = {k: r for k, r in recs.items()
                if r.get("execution_status") == "ok"}
    n_ev = len(evalable)
    succ = sum(r["success"] for r in evalable.values())
    osr = sum(r["osr"] for r in evalable.values())
    spl = sum(r.get("spl", 0.0) for r in evalable.values())
    ne = [r["final_dist"] for r in evalable.values()
          if r["final_dist"] is not None]
    steps = [r["steps"] for r in evalable.values()]
    by_split = {}
    for split in ("val_seen", "val_unseen"):
        s = [r for r in evalable.values() if r["split"] == split]
        if s:
            ss = sum(r["success"] for r in s)
            by_split[split] = {
                "n": len(s), "sr": ss / len(s),
                "osr": sum(r["osr"] for r in s) / len(s),
                "ne_mean": (sum(r["final_dist"] for r in s
                                if r["final_dist"] is not None)
                            / max(1, sum(1 for r in s
                                         if r["final_dist"] is not None))),
                "spl": sum(r.get("spl", 0.0) for r in s) / len(s),
                "mean_steps": sum(r["steps"] for r in s) / len(s)}
    groups = {}
    for g in ("short", "medium", "long"):
        s = [r for r in evalable.values() if r.get("length_group") == g]
        if s:
            groups[g] = {"n": len(s),
                         "sr": sum(r["success"] for r in s) / len(s)}
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
            "failures": dict(failures),
            "exceptions": sum(1 for r in recs.values()
                              if r.get("execution_status") != "ok")}


def per_episode_alignment(a: dict, b: dict, out_csv: Path,
                          name_a: str = "v4", name_b: str = "v3") -> dict:
    """Pairwise alignment of two journals on shared episode ids.

    Writes one CSV row per common episode, columns `<name_a>_*` and
    `<name_b>_*`, so the V4/V3 CSV has the same shape as the V4/V5 one.
    """
    common = sorted(set(a) & set(b))
    both_succ = sum(1 for e in common if a[e]["success"] and b[e]["success"])
    a_only = sum(1 for e in common if a[e]["success"] and not b[e]["success"])
    b_only = sum(1 for e in common if b[e]["success"] and not a[e]["success"])
    neither = len(common) - both_succ - a_only - b_only
    header = (["episode_id", "split", "map"]
              + [f"{name_a}_success", f"{name_b}_success",
                 f"{name_a}_final_dist", f"{name_b}_final_dist",
                 f"{name_a}_steps", f"{name_b}_steps",
                 f"{name_a}_spl", f"{name_b}_spl"])
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for e in common:
            ra, rb = a[e], b[e]
            w.writerow([e, ra["split"], ra["map"], ra["success"], rb["success"],
                        ra.get("final_dist"), rb.get("final_dist"),
                        ra["steps"], rb["steps"],
                        f"{ra.get('spl', 0):.4f}", f"{rb.get('spl', 0):.4f}"])
    return {"common": len(common), "both_success": both_succ,
            f"{name_a}_only_success": a_only, f"{name_b}_only_success": b_only,
            "neither": neither}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def render_md(s4: dict, s3: dict, align: dict, v3_ckpt: Path,
              v3_ckpt_sha: str, out_md: Path) -> None:
    vs, vu = s4["by_split"]["val_seen"], s4["by_split"]["val_unseen"]
    v3s, v3u = s3["by_split"]["val_seen"], s3["by_split"]["val_unseen"]
    hist = HISTORICAL_BASELINE["M-v3 (10E, old protocol)"]["val_sr"]
    lines = [
        "# V4 / V3（10D M-v3）修正协议闭环对比报告（val）",
        "",
        "## 1. 总表（统一协议：SR/OSR 20 m，NE=最终距离，SPL=成功率加权路径比，max steps=200）",
        "",
        "V3 = Stage-10D M-v3 checkpoint 在修正协议"
        "（corrected-history-pose-emptycand，与 V4/V5 官方评测同 manifest、"
        "同动作空间、同 20 m STOP 判据）下对全部 5,324 个 val 任务的重测。",
        "",
        "| 指标 | V4 (val) | V3 修正协议 (val) | 差异 |",
        "|---|---:|---:|---:|",
        f"| Val-Seen SR | {vs['sr']:.4f} | {v3s['sr']:.4f} | {v3s['sr']-vs['sr']:+.4f} |",
        f"| Val-Unseen SR | {vu['sr']:.4f} | {v3u['sr']:.4f} | {v3u['sr']-vu['sr']:+.4f} |",
        f"| Val OSR | {s4['osr']:.4f} | {s3['osr']:.4f} | {s3['osr']-s4['osr']:+.4f} |",
        f"| Val NE (m) | {s4['ne_mean']:.1f} | {s3['ne_mean']:.1f} | {s3['ne_mean']-s4['ne_mean']:+.1f} |",
        f"| Val SPL | {s4['spl']:.4f} | {s3['spl']:.4f} | {s3['spl']-s4['spl']:+.4f} |",
        "",
        "## 2. 逐 episode 对齐（5,324 共同任务）",
        "",
        f"- 共同任务：{align['common']}",
        f"- 双成功：{align['both_success']} | 仅 V4 成功：{align['v4_only_success']} | 仅 V3 成功：{align['v3_only_success']} | 双失败：{align['neither']}",
        "",
        "## 3. 分析",
        "",
        f"1. V4 vs V3（同为修正协议）：Val-Seen SR {vs['sr']:.4f} vs {v3s['sr']:.4f}"
        f"（{v3s['sr']-vs['sr']:+.4f}），Val-Unseen SR {vu['sr']:.4f} vs"
        f" {v3u['sr']:.4f}（{v3u['sr']-vu['sr']:+.4f}）。",
        f"2. V3 修正协议与历史 10E 旧协议（{hist}）的差异来自协议修正"
        f"（history 顺序、pose 相对编码、空候选表示），两数不可直接比较。",
        f"3. 失败构成 V3（修正协议）：{s3['failures']}",
        f"4. V3 异常任务数：{s3['exceptions']}",
        "",
        "## 4. 复现信息",
        "",
        "- 评测代码：experiments/v4_closed_loop_revalidation/scripts/run_closed_loop.py（--model v3）",
        f"- V3 checkpoint：{v3_ckpt}",
        f"- V3 checkpoint SHA256：{v3_ckpt_sha}",
        f"- 任务清单：Stage 06 val manifest（5,324 任务，manifest SHA 与 V4/V5 官方评测一致）",
        "- 历史 10E M-v3 结果（旧协议）仅作基线展示，不参与对齐统计",
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-ckpt",
                        default=str(REPO / "experiments/10_goal_query_mec_training"
                                    "/10D_transformer_m_training/results/runs_10d"
                                    "/seed42/best.pt"))
    args = parser.parse_args()
    v3_ckpt = Path(args.v3_ckpt)
    v3_sha = sha256(v3_ckpt)

    v4 = load(V4, "full_val")
    v3 = load(V3, "v3_val")
    s4, s3 = stats(v4, "v4"), stats(v3, "v3")
    OUT.mkdir(parents=True, exist_ok=True)
    align = per_episode_alignment(v4, v3, OUT / "v4_v3_alignment.csv")
    out = {"v4_val": s4, "v3_corrected_val": s3, "alignment": align,
           "v3_ckpt_sha": v3_sha,
           "historical_baselines": HISTORICAL_BASELINE,
           "protocol": "corrected-history-pose-emptycand",
           "v3_scope": "val 5,324 tasks (no test split run)"}
    (OUT / "v3_revalidation_metrics.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    render_md(s4, s3, align, v3_ckpt, v3_sha, OUT / "v3_comparison_report.md")
    print(json.dumps({"v3": {"n": s3["n_evalable"], "sr": s3["sr"],
                             "by_split": s3["by_split"]},
                      "alignment": align}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
