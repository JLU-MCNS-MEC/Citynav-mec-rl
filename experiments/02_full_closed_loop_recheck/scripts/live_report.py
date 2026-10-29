#!/usr/bin/env python3
"""Live experiment 02 reporter.

Watches the per-shard JSONL journals, appends every finished trajectory to
LIVE_RESULTS.md, and pushes the updated file to GitHub at most once per minute.
It is read-only with respect to the running evaluation and only stages
LIVE_RESULTS.md, so it cannot disturb the experiment branches.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "experiments/02_full_closed_loop_recheck"
RUNS = HERE / "runs"
MD = HERE / "LIVE_RESULTS.md"
FEED = HERE / "live_feed.log"
SUITE_LOG = HERE / "suite.log"
STATE = Path("/tmp/citynav_exp02_live_state.json")
BRANCH = "exp/visual-encoder-ablation-20261028"
CLASSES = ("Stop", "MoveForward", "TurnRight", "TurnLeft", "GoUp", "GoDown")
EXPECTED_RUNS = {
    "val_seen_small_roundA",
    "val_seen_transformer_m_roundA",
    "val_unseen_small_roundA",
    "val_unseen_transformer_m_roundA",
}
POLL_SECONDS = 20
PUSH_SECONDS = 60


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def _read_journals() -> list[dict]:
    found = []
    for path in sorted(RUNS.glob("*/*.jsonl")):
        run = path.parent.name
        shard = path.stem.rsplit("_shard", 1)[-1]
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "meta" in item:
                continue
            item["_run"] = run
            item["_shard"] = shard
            item["_key"] = f"{run}|{shard}|{item['episode_id']}"
            found.append(item)
    return found


def _aggregate(episodes: list[dict]) -> dict:
    if not episodes:
        return {}
    actions = Counter()
    for episode in episodes:
        for action in episode.get("action_sequence", []):
            actions[int(action)] += 1
    total_actions = max(sum(actions.values()), 1)
    distances = [float(e["step_distances_m"][-1]) for e in episodes]
    return {
        "n": len(episodes),
        "sr": sum(bool(e["success"]) for e in episodes) / len(episodes),
        "ne": sum(distances) / len(distances),
        "stopped": sum(bool(e["stopped"]) for e in episodes) / len(episodes),
        "entered": sum(bool(e["entered_20m"]) for e in episodes) / len(episodes),
        "left": sum(bool(e["left_after_entering_20m"]) for e in episodes) / len(episodes),
        "mean_steps": sum(int(e["steps"]) for e in episodes) / len(episodes),
        "actions": {
            name: actions.get(index, 0) / total_actions
            for index, name in enumerate(CLASSES)
        },
    }


def _render_md(state: dict) -> str:
    by_run: dict[str, list[dict]] = defaultdict(list)
    for episode in state.values():
        by_run[episode["_run"]].append(episode)
    lines = [
        "# Experiment 02 live results (500-step budget)",
        "",
        f"- updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- completed trajectories: **{len(state)}**",
        f"- expected per run: 24; runs: {len(by_run)} / {len(EXPECTED_RUNS)} started",
        "",
        "| run | done | SR | NE mean (m) | stop rate | entered 20m | left after entry | mean steps | actions (top 3) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in sorted(by_run):
        agg = _aggregate(by_run[run])
        top = sorted(agg["actions"].items(), key=lambda item: -item[1])[:3]
        top_text = ", ".join(f"{name} {value*100:.0f}%" for name, value in top)
        lines.append(
            f"| {run} | {agg['n']}/24 | {agg['sr']*100:.1f}% | "
            f"{agg['ne']:.1f} | {agg['stopped']*100:.1f}% | "
            f"{agg['entered']*100:.1f}% | {agg['left']*100:.1f}% | "
            f"{agg['mean_steps']:.1f} | {top_text} |"
        )
    lines += [
        "",
        "## Per-trajectory results (completion order)",
        "",
        "| # | time | run | map | episode | steps | stopped | success | NE (m) | min dist (m) | entered 20m | left after entry | actions top 3 |",
        "|---:|---|---|---|---|---:|---|---|---:|---:|---|---|---|",
    ]
    ordered = sorted(state.values(), key=lambda e: e.get("_seen_at", ""))
    for index, episode in enumerate(ordered, 1):
        counts = episode.get("action_counts", {})
        top = sorted(counts.items(), key=lambda item: -item[1])[:3]
        top_text = ", ".join(f"{name} {value}" for name, value in top)
        lines.append(
            f"| {index} | {episode.get('_seen_at','')} | {episode['_run']} | "
            f"{episode.get('map_name','')} | {episode['episode_id'][:12]} | "
            f"{episode['steps']} | {episode['stopped']} | {episode['success']} | "
            f"{episode['ne_m']:.1f} | {episode.get('min_dist_3d_m', float('nan')):.1f} | "
            f"{episode.get('entered_20m')} | {episode.get('left_after_entering_20m')} | "
            f"{top_text} |"
        )
    lines += [
        "",
        "Full per-step traces are in `runs/<run>/*.jsonl`. Test-Unseen is not used.",
        "",
    ]
    return "\n".join(lines)


def _git_push() -> None:
    try:
        subprocess.run(["git", "add", str(MD)], cwd=ROOT, check=False,
                       capture_output=True, text=True)
        subprocess.run(
            ["git", "commit", "-m", f"Live experiment 02 results ({_now()})", "--", str(MD)],
            cwd=ROOT, check=False, capture_output=True, text=True,
        )
        result = subprocess.run(
            ["git", "push", "origin", BRANCH],
            cwd=ROOT, check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[{_now()}] push failed: {result.stderr.strip()[:200]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[{_now()}] git error: {exc}", flush=True)


def main() -> int:
    state = _load_state()
    last_push = 0.0
    idle_polls = 0
    while True:
        episodes = _read_journals()
        new = 0
        for episode in episodes:
            key = episode["_key"]
            if key in state:
                continue
            episode["_seen_at"] = _now()
            state[key] = episode
            new += 1
            counts = episode.get("action_counts", {})
            top = sorted(counts.items(), key=lambda item: -item[1])[:3]
            top_text = ", ".join(f"{name}={value}" for name, value in top)
            line = (
                f"[{_now()}] {episode['_run']} episode={episode['episode_id'][:12]} "
                f"map={episode.get('map_name','')} steps={episode['steps']} "
                f"stopped={episode['stopped']} success={episode['success']} "
                f"NE={episode['ne_m']:.1f}m min={episode.get('min_dist_3d_m', float('nan')):.1f}m "
                f"entered20={episode.get('entered_20m')} actions={top_text}"
            )
            print(line, flush=True)
            with FEED.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        if new:
            _save_state(state)
            MD.write_text(_render_md(state), encoding="utf-8")
            idle_polls = 0
            now = time.time()
            if now - last_push >= PUSH_SECONDS:
                _git_push()
                last_push = now
        else:
            idle_polls += 1
        finished = SUITE_LOG.exists() and "experiment 02 suite finished" in SUITE_LOG.read_text(
            encoding="utf-8", errors="ignore"
        )
        if finished and idle_polls >= 2:
            if state:
                MD.write_text(_render_md(state), encoding="utf-8")
                _git_push()
            print(f"[{_now()}] experiment 02 live reporting finished", flush=True)
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
