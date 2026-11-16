#!/usr/bin/env python3
"""Start the DQN stage automatically once all 8 PPO training runs are finished.

The PPO evaluations may still be running; DQN training starts in parallel with
them, which is the requested parallel training/testing mode.  Baselines are a
separate stage and are NOT re-run here.

Launch once in the background:
  nohup python chain_dqn.py > results/scheduler/chain_dqn.out 2>&1 &
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P  # noqa: E402

SCHED = P.RES / "scheduler"
PPO_STATUS = SCHED / "ppo_status.json"
DQN_STATUS = SCHED / "dqn_status.json"
LOG = SCHED / "dqn_launch.out"


def ppo_training_done() -> bool:
    if not PPO_STATUS.exists():
        return False
    try:
        st = json.loads(PPO_STATUS.read_text())
    except json.JSONDecodeError:
        return False
    jobs = [j for j in st.get("jobs", {}).values() if j.get("kind") == "train"]
    return len(jobs) >= 8 and all(j.get("status") in ("done", "failed")
                                  for j in jobs)


def dqn_already_started() -> bool:
    if not DQN_STATUS.exists():
        return False
    try:
        st = json.loads(DQN_STATUS.read_text())
    except json.JSONDecodeError:
        return False
    return any(j.get("status") == "running" for j in st.get("jobs", {}).values())


def main() -> int:
    SCHED.mkdir(parents=True, exist_ok=True)
    print(f"[chain] waiting for PPO training to finish: {PPO_STATUS}",
          flush=True)
    while not ppo_training_done():
        time.sleep(60)
    print("[chain] PPO training complete; starting DQN", flush=True)
    if dqn_already_started():
        print("[chain] DQN already running; nothing to do", flush=True)
        return 0
    cmd = [sys.executable, str(HERE / "scheduler.py"), "--stage", "dqn",
           "--parallel-train", "4", "--parallel-eval", "4",
           "--episodes", "4000", "--report-every", "60"]
    with LOG.open("a", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True)
    print(f"[chain] DQN scheduler pid={p.pid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
