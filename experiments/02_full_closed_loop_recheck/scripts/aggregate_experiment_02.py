#!/usr/bin/env python3
"""Aggregate experiment 02 closed-loop runs into cap-sensitivity results.

Each run directory contains one <split>_<model>_full_summary.json produced by
aggregate_closed_loop.py, with per-episode step distance traces.  This script
computes SR / NE / truncation at the fixed 500-step cap, action output
distributions, and 20 m entry/exit/stop diagnostics.

No target or trajectory information is fed back into the model; all metrics are
evaluator-side only.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CAPS = (500,)
CLASSES = ("Stop", "MoveForward", "TurnRight", "TurnLeft", "GoUp", "GoDown")


def _distance_at(episode: dict, cap: int) -> float:
    trace = episode["step_distances_m"]
    index = min(int(cap), len(trace) - 1)
    return float(trace[index])


def _summarize(episodes: list[dict]) -> dict:
    out = {"n": len(episodes), "caps": {}}
    for cap in CAPS:
        distances = np.asarray([_distance_at(e, cap) for e in episodes])
        out["caps"][str(cap)] = {
            "success_rate": float(np.mean(distances <= 20.0)),
            "ne_m_mean": float(np.mean(distances)),
            "ne_m_median": float(np.median(distances)),
            "episodes_over_cap": int(sum(e["steps"] >= cap and not e["stopped"] for e in episodes)),
        }
    actions = Counter()
    for episode in episodes:
        for action in episode["action_sequence"]:
            actions[int(action)] += 1
    total = max(sum(actions.values()), 1)
    out["action_distribution"] = {
        name: {
            "count": int(actions.get(index, 0)),
            "fraction": float(actions.get(index, 0) / total),
        }
        for index, name in enumerate(CLASSES)
    }
    out["entered_20m"] = float(np.mean([e["entered_20m"] for e in episodes]))
    out["left_after_entering_20m"] = float(
        np.mean([e["left_after_entering_20m"] for e in episodes])
    )
    out["stopped_within_20m"] = float(
        np.mean([e["stopped_within_20m"] for e in episodes])
    )
    out["stop_rate"] = float(np.mean([e["stopped"] for e in episodes]))
    out["max_steps_reached"] = float(np.mean([e["max_steps_reached"] for e in episodes]))
    out["min_dist_3d_mean"] = float(np.mean([e["min_dist_3d_m"] for e in episodes]))
    out["min_dist_3d_min"] = float(np.min([e["min_dist_3d_m"] for e in episodes]))
    out["mean_steps"] = float(np.mean([e["steps"] for e in episodes]))
    out["mean_step_latency_s"] = float(
        np.mean([e["mean_step_latency_s"] for e in episodes if e["mean_step_latency_s"]])
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for path in sorted(args.runs_dir.glob("*/*_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", payload)
        episodes = payload.get("episodes") or summary.get("episodes")
        if not isinstance(episodes, list):
            continue
        split = summary["split"]
        model = summary["model"]
        round_name = path.parent.name.rsplit("round", 1)[-1]
        for episode in episodes:
            trace = episode.get("step_distances_m")
            if not trace:
                raise SystemExit(f"missing step trace in {path}")
            grouped[(split, model, round_name)].append(episode)

    if not grouped:
        raise SystemExit("no completed run summaries found")

    report: dict = {"caps": list(CAPS), "runs": {}, "pooled": {}}
    csv_rows = []
    for key, episodes in sorted(grouped.items()):
        split, model, round_name = key
        stats = _summarize(episodes)
        report["runs"][f"{split}/{model}/round{round_name}"] = stats
        for cap in CAPS:
            cap_stats = stats["caps"][str(cap)]
            csv_rows.append({
                "split": split,
                "model": model,
                "round": round_name,
                "cap": cap,
                "n": stats["n"],
                **cap_stats,
                "entered_20m": stats["entered_20m"],
                "left_after_entering_20m": stats["left_after_entering_20m"],
                "stopped_within_20m": stats["stopped_within_20m"],
                **{f"action_{name}": stats["action_distribution"][name]["fraction"]
                   for name in CLASSES},
            })

    for split in sorted({key[0] for key in grouped}):
        for model in sorted({key[1] for key in grouped}):
            episodes = []
            for key, value in grouped.items():
                if key[0] == split and key[1] == model:
                    episodes.extend(value)
            report["pooled"][f"{split}/{model}"] = _summarize(episodes)

    (args.out_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "cap_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "| split | model | cap | n | SR | NE mean (m) | truncated | entered 20m | left after entry | stopped ≤20m |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, stats in sorted(report["pooled"].items()):
        split, model = key.split("/")
        for cap in CAPS:
            cap_stats = stats["caps"][str(cap)]
            lines.append(
                f"| {split} | {model} | {cap} | {stats['n']} | "
                f"{cap_stats['success_rate']*100:.1f}% | "
                f"{cap_stats['ne_m_mean']:.1f} | "
                f"{cap_stats['episodes_over_cap']} | "
                f"{stats['entered_20m']*100:.1f}% | "
                f"{stats['left_after_entering_20m']*100:.1f}% | "
                f"{stats['stopped_within_20m']*100:.1f}% |"
            )
    (args.out_dir / "cap_sensitivity.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"wrote {args.out_dir/'summary.json'}, cap_sensitivity.csv/.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
