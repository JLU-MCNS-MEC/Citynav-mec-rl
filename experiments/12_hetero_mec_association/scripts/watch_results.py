#!/usr/bin/env python3
"""Emit one line per newly finished baseline / RL evaluation, plus progress.

Designed for the Monitor tool: stdout lines become notifications.  State of
already-reported files is kept in results/scheduler/seen.json so re-arming the
monitor does not repeat old results.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P  # noqa: E402

SEEN = P.RES / "scheduler" / "seen.json"
GROUP_ORDER = ("two_homo", "five_homo", "two_hetero", "five_hetero")
STRATEGIES = ("random", "max_rate", "max_quality", "oracle")
ALGOS = ("ppo", "dqn")
SEEDS = (42, 43)


def load_seen():
    if SEEN.exists():
        try:
            return set(json.loads(SEEN.read_text()))
        except json.JSONDecodeError:
            pass
    return set()


def save_seen(seen):
    SEEN.parent.mkdir(parents=True, exist_ok=True)
    tmp = SEEN.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(seen)))
    tmp.replace(SEEN)


def fmt(x, pct=False):
    if x is None:
        return "-"
    return f"{100*x:.2f}%" if pct else f"{x:.2f}"


def scan(seen):
    events = []
    for g in GROUP_ORDER:
        for s in STRATEGIES:
            p = P.RES / "baselines" / g / s / f"seed{P.EVAL_SEED}" / "summary.json"
            if p.exists() and str(p) not in seen:
                try:
                    d = json.loads(p.read_text())
                except json.JSONDecodeError:
                    continue
                seen.add(str(p))
                events.append(
                    f"RESULT baseline {g}/{s} n={d.get('n_ok')} "
                    f"SR={fmt(d.get('sr'), True)} E2E={fmt(d.get('e2e_mean_ms'))} "
                    f"P95={fmt(d.get('e2e_p95_ms'))} "
                    f"Inf={fmt(d.get('inf_mean_ms'))} "
                    f"Comm={fmt(d.get('comm_mean_ms'))} "
                    f"Enc={fmt(d.get('encode_mean_ms'))} "
                    f"share={json.dumps(d.get('model_step_share', {}))}")
        for algo in ALGOS:
            for seed in SEEDS:
                p = P.RES / "eval" / f"{g}_{algo}_seed{seed}" / "summary.json"
                if p.exists() and str(p) not in seen:
                    try:
                        d = json.loads(p.read_text())
                    except json.JSONDecodeError:
                        continue
                    seen.add(str(p))
                    events.append(
                        f"RESULT eval {g}/{algo}/s{seed} n={d.get('n')} "
                        f"SR={fmt(d.get('sr'), True)} "
                        f"E2E={fmt(d.get('e2e_mean_ms'))} "
                        f"P95={fmt(d.get('e2e_p95_ms'))} "
                        f"Inf={fmt(d.get('inf_mean_ms'))} "
                        f"share={json.dumps(d.get('model_step_share', {}))}")
    return events


def progress():
    parts = []
    for stage in ("baselines", "ppo", "dqn"):
        sp = P.RES / "scheduler" / f"{stage}_status.json"
        if not sp.exists():
            continue
        try:
            st = json.loads(sp.read_text())
        except json.JSONDecodeError:
            continue
        jobs = st.get("jobs", {})
        run = sum(1 for j in jobs.values() if j.get("status") == "running")
        done = sum(1 for j in jobs.values() if j.get("status") == "done")
        fail = sum(1 for j in jobs.values() if j.get("status") == "failed")
        evr = sum(1 for j in jobs.values()
                  if j.get("eval", {}).get("status") == "running")
        evd = sum(1 for j in jobs.values()
                  if j.get("eval", {}).get("status") == "done")
        # best-effort training progress
        ep = []
        for j in jobs.values():
            if j.get("status") == "running" and j.get("kind") == "train":
                pr = j.get("progress", {})
                if pr.get("episode"):
                    ep.append(pr["episode"])
        extra = ""
        if ep:
            extra = f" train_ep={min(ep)}-{max(ep)}"
        parts.append(f"{stage}: run={run} done={done} fail={fail} "
                     f"eval_run={evr} eval_done={evd}{extra}")
    return "PROGRESS " + " | ".join(parts)


def main() -> int:
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    progress_every = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    seen = load_seen()
    last_progress = time.time()
    # report anything that finished while no monitor was armed
    for e in scan(seen):
        print(e, flush=True)
    save_seen(seen)
    while True:
        time.sleep(interval)
        new = scan(seen)
        for e in new:
            print(e, flush=True)
        save_seen(seen)
        if time.time() - last_progress >= progress_every:
            print(progress(), flush=True)
            last_progress = time.time()


if __name__ == "__main__":
    raise SystemExit(main())
