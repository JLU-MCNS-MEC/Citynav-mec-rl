#!/usr/bin/env python3
"""Generate case_studies/index.md from the per-video metadata JSONs.

Never hand-edited: every number in the table comes from the metadata the
renderer wrote (which itself was computed from the trajectory JSONs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CASE_ROOT = REPO / "reports/v4_v5_comparison/case_studies"
CATEGORY_ORDER = ["v5_success_v4_failure", "v4_success_v3_failure",
                  "v4_success_v5_failure"]
CATEGORY_TITLES = {
    "v5_success_v4_failure": "A: V5 success / V4 failure",
    "v4_success_v3_failure": "B: V4 success / V3 failure",
    "v4_success_v5_failure": "C: V4 success / V5 failure",
}
CATEGORY_DIRS = {"A": "v5_success_v4_failure",
                 "B": "v4_success_v3_failure",
                 "C": "v4_success_v5_failure"}


def load_metas(meta_dir: Path) -> list[dict]:
    metas = []
    for p in sorted(meta_dir.glob("*.json")):
        if p.name.startswith("."):
            continue
        metas.append(json.loads(p.read_text(encoding="utf-8")))
    metas.sort(key=lambda m: (CATEGORY_ORDER.index(
        CATEGORY_DIRS.get(m["category"], m["category"]))
        if CATEGORY_DIRS.get(m["category"], m["category"]) in CATEGORY_ORDER
        else 99, m["episode_id"]))
    return metas


def build_index(meta_dir: Path, ctx: dict) -> str:
    metas = load_metas(meta_dir)
    lines = [
        "# V4 / V5 / V3 闭环导航对比视频案例集",
        "",
        f"共 {len(metas)} 个严格配对案例（同一 episode_id、同一任务、"
        "同一评测协议，轨迹逐 step 核对）。所有视频 1280×720 / H.264，"
        "目标标记仅为事后可视化，**不是策略输入**（visualization only — "
        "not policy input）。",
        "",
        "## 案例总表",
        "",
        "| Category | Episode | Split / Map | Difficulty | Winner | Loser | "
        "Loser failure | Video | Metadata |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in metas:
        eid = m["episode_id"]
        winner, loser = m["winner_model"], m["loser_model"]
        failure = m["models"][loser]["failure"]
        name = f"{eid.replace(':', '_')}_{winner}_vs_{loser}"
        cat_dir = CATEGORY_DIRS.get(m["category"], m["category"])
        rel_video = f"{cat_dir}/{name}.mp4"
        rel_meta = f"metadata/{name}.json"
        lines.append(
            f"| {CATEGORY_TITLES.get(m['category'], m['category'])} | {eid} | "
            f"{m['split']} / {m['map']} | {m['difficulty']} | {winner} | "
            f"{loser} | {failure} | [{name}.mp4]({rel_video}) | "
            f"[{name}.json]({rel_meta}) |")
    lines += [
        "",
        "## 评测协议（所有案例共用）",
        "",
        f"- 协议：{ctx.get('protocol', 'corrected-history-pose-emptycand')}",
        f"- 成功判据：STOP 后最终距离 ≤ {ctx.get('stop_radius_m', 20)} m，"
        f"步数上限 {ctx.get('step_cap', 200)}；动作空间 "
        "Stop/Forward/TurnLeft/TurnRight（前进 5 m，转弯 30°）",
        "- 动作概率：由 replay_corrected.py 用各模型 checkpoint 对已存轨迹"
        "逐步重算的真实 softmax（argmax 与已存动作逐步一致，不一致即拒绝渲染）",
        "- 信念展示：画面中的估计点来自轨迹 JSON 每步 `estimate` 字段"
        "（09C 连续信念均值），非事后拟合",
        "- 时间同步：帧按两轨迹较长者推进；已结束模型冻结于末步并加 "
        "ENDED 徽标，绝不虚构轨迹",
        f"- V3 说明：{ctx.get('v3_scope', '')}",
        "",
        "## 案例选择原则",
        "",
        "- 仅用逐 episode 对齐 CSV 自动筛选（A/C 来自 "
        "per_episode_alignment.csv，B 来自 v4_v3_alignment.csv）",
        "- 严格配对：winner/loser 的 map、start、target 必须完全一致，"
        "否则剔除（selection.json 的 notes 记录每一例剔除原因）",
        "- 优先：失败原因多样、胜者信念收敛（末步估计误差 ≤ 8 m 且估计点距"
        "目标 ≤ 20 m）、败者定位漂移（最大估计误差 ≥ 30 m）、步数"
        " [8,160]、地图不重复",
        "",
        "## 配套文件",
        "",
        "- `selection.json` — 筛选结果与理由",
        "- `metadata/<case>.json` — 每视频元数据（checkpoint SHA256、轨迹"
        "JSON 引用与 sha256、指标上下文、结论、生成脚本 sha256 与参数）",
        "- `metadata/replay/` — 逐步动作概率重算结果",
        "- `metadata/generation/` — 生成脚本冻结副本 + render_params.json",
        "- `covers/` — 每视频封面",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-dir",
                        default=str(CASE_ROOT / "metadata"))
    parser.add_argument("--out", default=str(CASE_ROOT / "index.md"))
    args = parser.parse_args()
    ctx = {"v3_scope": "V3 为 10D M-v3 在修正协议下对全部 5,324 个 val "
                      "任务的重测（无 test 分片）；历史 10E 旧协议结果不参与"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_index(Path(args.meta_dir), ctx), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
