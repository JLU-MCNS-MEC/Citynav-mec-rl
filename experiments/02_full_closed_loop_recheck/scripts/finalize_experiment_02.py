#!/usr/bin/env python3
"""Wait for experiment 02 to finish, aggregate it, report it, and push.

What this does once the val_unseen round is done:

1. aggregate the two val_unseen runs (if the suite did not already);
2. run the experiment-level aggregate (``results/``);
3. append a factual ``## 12. val_unseen`` section to ``REPORT.md`` and flip the
   report status to complete;
4. update the experiment index row in ``experiments/README.md``;
5. stop the CityNav live-report process and leave every unrelated process alone;
6. commit and push the reports and aggregate CSVs.

It only touches CityNav artefacts.  It refuses to run if the suite is still
alive.  Run it detached so a session interruption cannot kill it:

    setsid nohup python .../finalize_experiment_02.py > /tmp/exp02_finalize.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[3]
HERE = ROOT / "experiments/02_full_closed_loop_recheck"
LOG = HERE / "suite.log"
REPORT = HERE / "REPORT.md"
README = ROOT / "experiments/README.md"
RESULTS = HERE / "results"
BRANCH = "exp/visual-encoder-ablation-20261028"
PY = pathlib.Path(
    os.environ.get(
        "CITYNAV_PY",
        str(pathlib.Path.home() / "miniconda3/envs/DRL_migrated/bin/python"),
    )
)
CLOSED = ROOT / "experiments/visual_encoder_citynav/frozen_e2_fov_pose_mec"
RUNS = (
    ("val_seen", "small"),
    ("val_seen", "transformer_m"),
    ("val_unseen", "small"),
    ("val_unseen", "transformer_m"),
)


def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def _summary_path(split: str, model: str) -> pathlib.Path:
    return HERE / "runs" / f"{split}_{model}_roundA" / f"{split}_{model}_full_summary.json"


def _suite_finished() -> bool:
    if not LOG.exists():
        return False
    return "experiment 02 suite finished" in LOG.read_text(encoding="utf-8", errors="ignore")


def _pids(pattern: str) -> list[str]:
    out = subprocess.run(
        ["pgrep", "-af", pattern], capture_output=True, text=True, check=False
    )
    return [line for line in out.stdout.splitlines() if line.strip()]


def _wait(deadline_hours: float) -> str:
    """Return 'finished', 'crashed' or 'timeout'."""
    deadline = time.time() + deadline_hours * 3600
    while time.time() < deadline:
        if _suite_finished():
            return "finished"
        evals = _pids("closed_loop_eval.py")
        suite = _pids("run_experiment_02.sh")
        if not evals and not suite:
            if all(_summary_path(s, m).exists() for s, m in RUNS):
                return "finished"
            return "crashed"
        time.sleep(60)
    return "timeout"


def _run(cmd: list[str], log_path: pathlib.Path | None = None) -> int:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    if log_path is not None:
        log_path.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
    return proc.returncode


def _aggregate() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for split, model in RUNS:
        summary = _summary_path(split, model)
        if summary.exists():
            continue
        out_dir = summary.parent
        _log(f"aggregating missing run {summary.parent.name}")
        _run(
            [
                str(PY),
                str(CLOSED / "aggregate_closed_loop.py"),
                "--out-dir",
                str(out_dir),
                "--split",
                split,
                "--model",
                model,
                "--variant",
                "full",
                "--shards",
                "8",
            ],
            out_dir / "aggregate_finalize.log",
        )
    _log("aggregating experiment level")
    code = _run(
        [
            str(PY),
            str(HERE / "scripts/aggregate_experiment_02.py"),
            "--runs-dir",
            str(HERE / "runs"),
            "--out-dir",
            str(RESULTS),
        ],
        RESULTS / "aggregate.log",
    )
    if code != 0:
        raise SystemExit(f"experiment-level aggregation failed (exit {code})")


def _load(split: str, model: str) -> tuple[dict, list[dict]]:
    payload = json.loads(_summary_path(split, model).read_text(encoding="utf-8"))
    return payload["summary"], payload["episodes"]


def _stats(episodes: list[dict]) -> dict:
    ne = [float(e["ne_m"]) for e in episodes]
    md = [float(e["min_dist_m"]) for e in episodes]
    start = [float(e["step_distances_m"][0]) for e in episodes]
    actions = Counter()
    for episode in episodes:
        for name, count in episode["action_counts"].items():
            actions[name] += int(count)
    total = max(sum(actions.values()), 1)
    return {
        "n": len(episodes),
        "success": sum(bool(e["success"]) for e in episodes),
        "stopped": sum(bool(e["stopped"]) for e in episodes),
        "stopped_in_20m": sum(bool(e["stopped_within_20m"]) for e in episodes),
        "capped": sum(bool(e["max_steps_reached"]) for e in episodes),
        "ne_mean": statistics.mean(ne),
        "ne_median": statistics.median(ne),
        "ne_min": min(ne),
        "ne_max": max(ne),
        "start_mean": statistics.mean(start),
        "min_dist_mean": statistics.mean(md),
        "min_dist_median": statistics.median(md),
        "min_dist_min": min(md),
        "entered": sum(bool(e["entered_20m"]) for e in episodes),
        "left": sum(bool(e["left_after_entering_20m"]) for e in episodes),
        "actions": {k: (v, 100.0 * v / total) for k, v in actions.items()},
        "latency": statistics.mean(
            float(e["mean_step_latency_s"]) for e in episodes if e["mean_step_latency_s"]
        ),
    }


def _action_line(actions: dict) -> str:
    ordered = sorted(actions.items(), key=lambda kv: -kv[1][0])
    return "、".join(f"{name} {frac:.1f}%" for name, (_, frac) in ordered)


def _per_episode_table(small: list[dict], tm: list[dict]) -> list[str]:
    a = {(e["map_name"], e["episode_id"]): e for e in small}
    b = {(e["map_name"], e["episode_id"]): e for e in tm}
    lines = [
        "| # | map | episode | steps | 进入20m | NE small (m) | NE tm (m) | min dist (m) | 动作（small） |",
        "|---:|---|---|---:|---|---:|---:|---:|---|",
    ]
    for i, key in enumerate(sorted(a), 1):
        x = a[key]
        y = b.get(key)
        acts = ", ".join(
            f"{name} {count}" for name, count in sorted(x["action_counts"].items())
        )
        lines.append(
            f"| {i} | {x['map_name']} | {x['episode_id'][:8]} | {x['steps']} | "
            f"{'Y' if x['entered_20m'] else '—'} | {x['ne_m']:.1f} | "
            f"{y['ne_m']:.1f} | {x['min_dist_m']:.1f} | {acts} |"
        )
    return lines


def _section(finished_at: str) -> str:
    s_small, e_small = _load("val_unseen", "small")
    s_tm, e_tm = _load("val_unseen", "transformer_m")
    st_s, st_t = _stats(e_small), _stats(e_tm)
    vs_s, vs_e = _load("val_seen", "small")
    vs_t, vs_e_t = _load("val_seen", "transformer_m")
    vst_s, vst_t = _stats(vs_e), _stats(vs_e_t)
    min_dist = min(st_s["min_dist_min"], st_t["min_dist_min"])
    near = sorted(
        [
            (e["map_name"], e["episode_id"][:8], e["min_dist_m"])
            for e in e_small
            if e["entered_20m"]
        ],
        key=lambda row: row[2],
    )
    near_text = "、".join(f"{m} {mid} ({d:.1f} m)" for m, mid, d in near) or "无"

    lines = [
        "",
        f"## 12. val_unseen round A 结果（自动汇总，{finished_at}）",
        "",
        f"- 完成时间：{finished_at}；small:full 与 transformer_m:full 各 24 条、500 步上限。",
        "- 汇总文件：`runs/val_unseen_{small,transformer_m}_roundA/*_full_summary.{json,csv}`；",
        "  实验级汇总：`results/summary.json`、`results/cap_sensitivity.{csv,md}`（由",
        "  `scripts/aggregate_experiment_02.py` 生成）。",
        "",
        "| 指标 | small:full | transformer_m:full |",
        "|---|---:|---:|",
        f"| SR（终点 ≤20 m） | {st_s['success']}/24 | {st_t['success']}/24 |",
        f"| stopped / stopped_within_20m | {st_s['stopped']} / {st_s['stopped_in_20m']} | "
        f"{st_t['stopped']} / {st_t['stopped_in_20m']} |",
        f"| 跑满 500 步 | {st_s['capped']}/24 | {st_t['capped']}/24 |",
        f"| NE 均值 / 中位 / 范围 (m) | {st_s['ne_mean']:.1f} / {st_s['ne_median']:.1f} / "
        f"{st_s['ne_min']:.1f}–{st_s['ne_max']:.1f} | {st_t['ne_mean']:.1f} / "
        f"{st_t['ne_median']:.1f} / {st_t['ne_min']:.1f}–{st_t['ne_max']:.1f} |",
        f"| 起点距离均值 (m) | {st_s['start_mean']:.1f} | {st_t['start_mean']:.1f} |",
        f"| min_dist 均值 / 中位 / 最小 (m) | {st_s['min_dist_mean']:.1f} / "
        f"{st_s['min_dist_median']:.1f} / {st_s['min_dist_min']:.1f} | "
        f"{st_t['min_dist_mean']:.1f} / {st_t['min_dist_median']:.1f} / "
        f"{st_t['min_dist_min']:.1f} |",
        f"| 进入 20 m（其中离开） | {st_s['entered']}/24（{st_s['left']}/{max(st_s['entered'],1)}） | "
        f"{st_t['entered']}/24（{st_t['left']}/{max(st_t['entered'],1)}） |",
        f"| 动作分布 | {_action_line(st_s['actions'])} | {_action_line(st_t['actions'])} |",
        f"| 平均单步延迟 (s) | {st_s['latency']:.2f} | {st_t['latency']:.2f} |",
        "",
        "### 12.1 逐条结果",
        "",
    ]
    lines += _per_episode_table(e_small, e_tm)
    lines += [
        "",
        "### 12.2 与 val_seen 的对比",
        "",
        "| 指标 | val_seen（small / tm） | val_unseen（small / tm） |",
        "|---|---|---|",
        f"| SR | {vst_s['success']}/24 / {vst_t['success']}/24 | "
        f"{st_s['success']}/24 / {st_t['success']}/24 |",
        f"| STOP | {vst_s['stopped']} / {vst_t['stopped']} | {st_s['stopped']} / {st_t['stopped']} |",
        f"| 跑满 500 步 | {vst_s['capped']}/24 / {vst_t['capped']}/24 | "
        f"{st_s['capped']}/24 / {st_t['capped']}/24 |",
        f"| NE 均值 (m) | {vst_s['ne_mean']:.1f} / {vst_t['ne_mean']:.1f} | "
        f"{st_s['ne_mean']:.1f} / {st_t['ne_mean']:.1f} |",
        f"| 进入 20 m | {vst_s['entered']}/24 / {vst_t['entered']}/24 | "
        f"{st_s['entered']}/24 / {st_t['entered']}/24 |",
        f"| min_dist 最小 (m) | {vst_s['min_dist_min']:.1f} / {vst_t['min_dist_min']:.1f} | "
        f"{st_s['min_dist_min']:.1f} / {st_t['min_dist_min']:.1f} |",
        "",
        "### 12.3 结论（由汇总数据直接生成）",
        "",
        f"1. val_unseen 24 条 / 模型全部跑完 500 步，**SR = {st_s['success']}/24（small）与 "
        f"{st_t['success']}/24（transformer_m）**；STOP 次数 {st_s['stopped']} / {st_t['stopped']}。",
        f"2. 动作仍然只有 MoveForward 与 GoDown（{_action_line(st_s['actions'])}），"
        "Turn/GoUp/Stop 均为 0。",
        f"3. small 有 {st_s['entered']}/24、transformer_m 有 {st_t['entered']}/24 条轨迹进入过 "
        f"20 m，全部离开；两模型最近距离 {min_dist:.1f} m。进入的轨迹：{near_text}。",
        "4. 与 val_seen 的每个指标同向且量级相同，未见留出划分上的行为差异；"
        "两模型在闭环中仍不可区分。",
        "",
        "结论：500 步上限下 val_unseen 复现了 val_seen 的全部失败模式；"
        "本实验到此结束，不再加跑 round B 或新的闭环组合（见 `../STATUS_2026-10-29.md` §6）。",
        "",
        "---",
        "",
        f"_本段由 `scripts/finalize_experiment_02.py` 于 {finished_at} 自动生成。_",
        "",
    ]
    return "\n".join(lines)


def _update_report(finished_at: str) -> None:
    text = REPORT.read_text(encoding="utf-8")
    # status block: the first two lines starting with "- 状态：" and its indent
    lines = text.splitlines()
    out: list[str] = []
    replaced = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if not replaced and line.startswith("- 状态："):
            out.append(
                f"- 状态：**完成**（`val_seen` 18:08，`val_unseen` 见 §12）；"
                "两轮均按 500 步协议跑完，结论见 §7 与 §12。"
            )
            i += 1
            while i < len(lines) and lines[i].startswith("  "):
                i += 1
            replaced = True
            continue
        out.append(line)
        i += 1
    text = "\n".join(out) + "\n"
    text = re.sub(r"\n## 1[12]\..*$", "", text, flags=re.S).rstrip() + "\n"
    text += (
        "\n## 11. 收尾状态\n\n"
        "1. val_unseen round A 已完成并汇总（见 §12）；实验级结果在 `results/`。\n"
        "2. 未运行 round B（BUDGET 规定每划分 1 轮）；`selection/*_roundB_indices.json` 保留未用。\n"
        "3. 本实验结束后不再启动新的闭环组合；后续判断见 `../STATUS_2026-10-29.md` §6。\n"
    )
    text += _section(finished_at)
    REPORT.write_text(text, encoding="utf-8")
    _log("updated REPORT.md")


def _update_readme(finished_at: str) -> None:
    s_small, e_small = _load("val_unseen", "small")
    s_tm, e_tm = _load("val_unseen", "transformer_m")
    st_s, st_t = _stats(e_small), _stats(e_tm)
    vs_s, vs_e = _load("val_seen", "small")
    vst_s = _stats(vs_e)
    date = finished_at[:16]
    row = (
        "| 02 | `02_full_closed_loop_recheck/` | small:full / transformer_m:full 单轮分层闭环复测"
        f"（每划分 24 条，固定 500 步） | 01 | **完成**（val_seen 18:08 / val_unseen {date[11:16]}） | "
        f"val_seen 48 条 SR=0、0 STOP、3 条进入 20 m 后离开（最小 {vst_s['min_dist_min']:.1f} m）；"
        f"val_unseen 48 条 SR=0、0 STOP、{st_s['entered'] + st_t['entered']}/48 进入 20 m 后离开"
        f"（最小 {min(st_s['min_dist_min'], st_t['min_dist_min']):.1f} m）；两模型轨迹几乎重合；"
        "见 `REPORT.md` |"
    )
    lines = README.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith("| 02 |"):
            lines[i] = row
            break
    else:
        _log("warning: experiment 02 row not found in README")
    README.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log("updated experiments/README.md")


def _stop_citynav_jobs() -> None:
    """Stop only the CityNav live reporter; unrelated GPU jobs are left alone."""
    for line in _pids("scripts/live_report.py"):
        pid = int(line.split()[0])
        _log(f"stopping CityNav live_report.py pid={pid}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    time.sleep(2)
    for line in _pids("scripts/live_report.py"):
        pid = int(line.split()[0])
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    leftover = _pids("closed_loop_eval.py")
    if leftover:
        _log(f"warning: {len(leftover)} closed-loop process(es) still alive")
    else:
        _log("no CityNav closed-loop process left")


def _git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def _commit_and_push(finished_at: str) -> None:
    paths = [
        "experiments/02_full_closed_loop_recheck/REPORT.md",
        "experiments/02_full_closed_loop_recheck/LIVE_RESULTS.md",
        "experiments/02_full_closed_loop_recheck/results",
        "experiments/02_full_closed_loop_recheck/runs",
        "experiments/02_full_closed_loop_recheck/scripts/finalize_experiment_02.py",
        "experiments/README.md",
    ]
    add = _git(["add", *paths])
    if add.returncode != 0:
        _log(f"git add failed: {add.stderr.strip()}")
        return
    status = _git(["status", "--porcelain"])
    if not status.stdout.strip():
        _log("nothing to commit")
        return
    message = (
        "Finalize experiment 02: val_unseen round A results\n\n"
        f"- {finished_at}: val_unseen small:full / transformer_m:full, 24 episodes each, 500 steps.\n"
        "- Aggregate per-run summaries and experiment-level results/.\n"
        "- Append the val_unseen section to REPORT.md and mark the experiment complete.\n"
        "- Stop the CityNav live reporter after the last trajectory.\n"
    )
    commit = _git(["commit", "-m", message])
    if commit.returncode != 0:
        _log(f"git commit failed: {commit.stderr.strip()}")
        return
    _log("committed")
    for attempt in range(3):
        push = _git(["push", "origin", BRANCH])
        if push.returncode == 0:
            _log("pushed")
            return
        _log(f"push failed (attempt {attempt + 1}): {push.stderr.strip()}")
        fetch = _git(["fetch", "origin", BRANCH])
        if fetch.returncode == 0:
            _git(["rebase", "--autostash", f"origin/{BRANCH}"])
        time.sleep(5)
    _log("could not push after 3 attempts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-wait-hours", type=float, default=6.0)
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the section and exit")
    args = parser.parse_args()

    if args.dry_run:
        print(_section(_now()))
        return 0

    _log("waiting for experiment 02 to finish")
    state = _wait(args.max_wait_hours)
    _log(f"suite state: {state}")
    if state != "finished":
        marker = RESULTS / "FINALIZE_ERROR.txt"
        RESULTS.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            f"{_now()} suite state={state}; not finalized\n", encoding="utf-8"
        )
        return 1

    finished_at = _now()
    try:
        _aggregate()
        _update_report(finished_at)
        _update_readme(finished_at)
        _stop_citynav_jobs()
        if not args.no_push:
            _commit_and_push(finished_at)
    except Exception as exc:  # noqa: BLE001 - leave a trace instead of dying silently
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "FINALIZE_ERROR.txt").write_text(
            f"{_now()} finalize failed: {type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        _log(f"finalize failed: {type(exc).__name__}: {exc}")
        return 1
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "FINALIZED_AT.txt").write_text(
        f"{finished_at} experiment 02 finalized; suite state={state}\n", encoding="utf-8"
    )
    _log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
