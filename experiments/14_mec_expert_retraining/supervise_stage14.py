#!/usr/bin/env python3
"""Stage 14-R1 supervisor (auto-managed).

Policy (user-approved 2026-11-17): training stays strictly one model at a time,
but each model's Val-Seen closed-loop evaluation overlaps the NEXT model's
training.  Sequence:

  wait Small training (already running) ->
  [Small eval || Medium train] -> [Medium eval || TS train] ->
  [TS eval || TM train] -> [TM eval || Large train] -> Large eval -> done

Every model: strict best.pt reload check, 2498-task closed loop, report update.
Any failure stops the supervisor (and is logged as FAILED).
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
KEYS = ["small", "medium", "ts", "tm", "large"]
NAMES = {"small": "Small", "medium": "Medium", "ts": "Transformer-S",
         "tm": "Transformer-M", "large": "Large"}
PY_TRAIN = os.environ.get(
    "PY_TRAIN", "/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python")
PY_EVAL = os.environ.get(
    "PY_EVAL", "/mnt/newhome/venvs/citynav-gpu/bin/python")
MANIFEST = str(REPO / "experiments/06_closed_loop_success_rate/results/seed42/"
                      "manifest.json")
TASKS = "/home/jlu-mcns/Workspace/zhangty/task_status/val_seen_ids.txt"
LOG = HERE / "logs"
N_TASKS = 2498

# The supervisor must export the data root itself: subprocesses inherit it.
os.environ.setdefault("CITYNAV_DATA", "/home/20261024_1/Workspace/DATA")
if not Path(os.environ["CITYNAV_DATA"]).is_dir():
    raise SystemExit(f"CITYNAV_DATA is not a directory: "
                     f"{os.environ['CITYNAV_DATA']}")

LOG.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with (LOG / "supervisor.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    with (LOG / "queue.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def pgrep(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern],
                          capture_output=True).returncode == 0


def paths(key: str) -> tuple[Path, Path, Path]:
    name = NAMES[key]
    return (HERE / "checkpoints" / name,
            HERE / "results" / name,
            HERE / "results" / name / "closed_loop")


def report() -> None:
    subprocess.run([PY_EVAL, str(HERE / "progress_report.py")],
                   stdout=open(LOG / "report.log", "a"),
                   stderr=subprocess.STDOUT)


def periodic_report(stop: threading.Event) -> None:
    while not stop.wait(300):
        report()


def wait_training(key: str) -> int:
    """Block until no train_expert process for key remains."""
    pat = f"train_expert.py --model {key}"
    while pgrep(pat):
        time.sleep(30)
    return 0


class WaitingProc:
    """Attach to a training process started by a previous supervisor."""

    def __init__(self, pattern: str):
        self.pattern = pattern

    def wait(self) -> int:
        while pgrep(self.pattern):
            time.sleep(30)
        return 0


def start_train(key: str):
    ckpt_dir, _, _ = paths(key)
    last = ckpt_dir / "last.pt"
    if (ckpt_dir / "best.pt").exists():
        log(f"{key}: best.pt exists, training skipped")
        return None
    pat = f"train_expert.py --model {key}"
    if pgrep(pat):
        log(f"{key}: training already running; attaching (no duplicate launch)")
        return WaitingProc(pat)
    cmd = [PY_TRAIN, str(HERE / "train_expert.py"), "--model", key,
           "--seed", "42", "--epochs", "20", "--patience", "10",
           "--device", "cuda"]
    if last.exists():
        cmd.append("--resume")
    log(f"{key}: train start ({'resume' if last.exists() else 'from-scratch'})")
    fh = open(LOG / f"train_{key}.log", "a")
    return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=str(REPO))


def verify_ckpt(key: str) -> bool:
    ckpt_dir, _, _ = paths(key)
    ckpt = ckpt_dir / "best.pt"
    if not ckpt.exists():
        log(f"{key}: best.pt missing")
        return False
    r = subprocess.run([PY_EVAL, str(HERE / "verify_ckpt.py"), "--model", key,
                        "--ckpt", str(ckpt)],
                       stdout=open(LOG / f"verify_{key}.log", "a"),
                       stderr=subprocess.STDOUT)
    log(f"{key}: checkpoint verify {'OK' if r.returncode == 0 else 'FAILED'}")
    return r.returncode == 0


def start_eval(key: str) -> subprocess.Popen:
    ckpt_dir, _, cloop = paths(key)
    cloop.mkdir(parents=True, exist_ok=True)
    cmd = [PY_EVAL, str(HERE / "eval_val_seen.py"), "--model", key,
           "--ckpt", str(ckpt_dir / "best.pt"), "--manifest", MANIFEST,
           "--task-file", TASKS, "--out-dir", str(cloop),
           "--step-cap", "200", "--device", "cuda"]
    log(f"{key}: closed-loop start (2498 tasks)")
    fh = open(LOG / f"eval_{key}.log", "a")
    return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=str(REPO))


def eval_complete(key: str) -> bool:
    _, _, cloop = paths(key)
    p = cloop / "val_seen_summary.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return bool(d.get("complete")) and int(d.get("completed_episodes", 0)) \
            >= N_TASKS
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    stop = threading.Event()
    threading.Thread(target=periodic_report, args=(stop,), daemon=True).start()
    log(f"supervisor start (pid {os.getpid()}) "
        f"git={subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()}")
    (LOG / "supervisor.pid").write_text(str(os.getpid()) + "\n")
    report()

    # Small training is already running under the previous launcher.
    if pgrep("train_expert.py --model small"):
        log("waiting for the running Small training to finish")
    wait_training("small")
    log("Small training finished")

    for i, key in enumerate(KEYS):
        if not verify_ckpt(key):
            log(f"{key}: CHECKPOINT VERIFY FAILED, stopping")
            stop.set()
            return 1
        ev = start_eval(key)
        tr = start_train(KEYS[i + 1]) if i + 1 < len(KEYS) else None
        if tr is not None:
            log(f"{KEYS[i+1]}: training runs concurrently with {key} eval")
        ev_rc = ev.wait()
        log(f"{key}: closed-loop rc={ev_rc}")
        if ev_rc != 0 or not eval_complete(key):
            log(f"{key}: EVAL FAILED/INCOMPLETE, stopping")
            stop.set()
            return 1
        report()
        log(f"{key}: MILESTONE complete (eval + report)")
        if tr is not None:
            tr_rc = tr.wait()
            nxt = KEYS[i + 1]
            log(f"{nxt}: train rc={tr_rc}")
            ckpt_dir, _, _ = paths(nxt)
            if tr_rc != 0 or not (ckpt_dir / "best.pt").exists():
                log(f"{nxt}: TRAIN FAILED, stopping")
                stop.set()
                return 1
    log("ALL_MODELS_DONE")
    stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
