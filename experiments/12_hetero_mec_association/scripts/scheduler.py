#!/usr/bin/env python3
"""Stage 12 job scheduler: parallel training + auto-evaluation + live status.

Stages:
  baselines : 4 groups x 4 strategies (fresh closed loop, seed 42)
  ppo       : 4 groups x 2 seeds (42, 43)
  dqn       : 4 groups x 2 seeds (42, 43)

Ordering: homogeneous groups first (two_homo, five_homo), then heterogeneous
(two_hetero, five_hetero).  Every training job is evaluated on Val-Seen as soon
as it finishes.

Status is written to results/scheduler/<stage>_status.json.  Use
  python scheduler.py --stage ppo --status
to print the live table.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P  # noqa: E402

GROUP_ORDER = ("two_homo", "five_homo", "two_hetero", "five_hetero")
RL_SEEDS = (42, 43)
PY = sys.executable
SCHED_DIR = P.RES / "scheduler"


def job_id(stage, group, tag, seed):
    return f"{stage}:{group}:{tag}:s{seed}"


def build_jobs(stage):
    jobs = []
    if stage == "baselines":
        for g in GROUP_ORDER:
            for s in P.STRATEGIES:
                jobs.append({
                    "id": job_id(stage, g, s, P.EVAL_SEED), "kind": "baseline",
                    "group": g, "tag": s, "seed": P.EVAL_SEED,
                    "out": f"baselines/{g}/{s}/seed{P.EVAL_SEED}"})
    else:
        for g in GROUP_ORDER:
            for seed in RL_SEEDS:
                jobs.append({
                    "id": job_id(stage, g, stage, seed), "kind": "train",
                    "group": g, "tag": stage, "seed": seed,
                    "out": f"rl/{g}/{stage}/seed{seed}",
                    "eval_out": f"eval/{g}_{stage}_seed{seed}"})
    return jobs


def load_status(path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {"stage": None, "jobs": {}, "started": time.time()}


def save_status(path, st):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, path)


def history_progress(run_dir, episodes):
    hp = run_dir / "history.jsonl"
    if not hp.exists():
        return {}
    last = None
    for line in hp.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = line
    if last is None:
        return {}
    try:
        rec = json.loads(last)
    except json.JSONDecodeError:
        return {}
    ep = int(rec.get("episode", 0)) + 1
    elapsed = float(rec.get("elapsed_s", 0.0)) or 1e-9
    rate = ep / elapsed
    eta = max(0.0, (episodes - ep) / rate) if rate > 0 else None
    out = {"episode": ep, "episodes": episodes,
           "elapsed_s": round(elapsed, 1),
           "eta_s": None if eta is None else round(eta, 1),
           "ma100_e2e_ms": rec.get("ma100_e2e_ms"),
           "ma100_reward": rec.get("ma100_reward"),
           "ma100_success": rec.get("ma100_success")}
    best = run_dir / "best_meta.json"
    if best.exists():
        try:
            out.update(json.loads(best.read_text()))
        except json.JSONDecodeError:
            pass
    dev = run_dir / "dev.jsonl"
    if dev.exists():
        lines = [l for l in dev.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        if lines:
            try:
                d = json.loads(lines[-1])
                out["dev_e2e_ms"] = d.get("e2e_mean_ms")
                out["dev_sr"] = d.get("sr")
            except json.JSONDecodeError:
                pass
    return out


def eval_progress(out_dir, total=P.EVAL_TOTAL if hasattr(P, "EVAL_TOTAL") else 2498):
    j = out_dir / "journal.jsonl"
    n = 0
    if j.exists():
        n = sum(1 for line in j.read_text(encoding="utf-8").splitlines()
                if line.strip())
    s = out_dir / "summary.json"
    res = None
    if s.exists():
        try:
            res = json.loads(s.read_text())
        except json.JSONDecodeError:
            pass
    return {"done": n, "total": total, "summary": res}


def fmt_eta(sec):
    if sec is None:
        return "-"
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


def table(st, stage):
    rows = []
    for jid, j in st["jobs"].items():
        line = {"id": jid, "status": j["status"]}
        if j["kind"] == "train":
            prog = j.get("progress", {})
            line["ep"] = f"{prog.get('episode','-')}/{prog.get('episodes','-')}"
            line["ma100_e2e"] = (f"{prog['ma100_e2e_ms']:.2f}"
                                 if prog.get("ma100_e2e_ms") is not None else "-")
            line["best_dev"] = (f"{prog.get('best_dev_e2e_ms'):.2f}"
                                if prog.get("best_dev_e2e_ms") is not None else "-")
            line["eta"] = fmt_eta(prog.get("eta_s"))
            ev = j.get("eval", {})
            line["eval"] = ev.get("status", "-")
            if ev.get("summary"):
                line["SR"] = f"{ev['summary']['sr']*100:.2f}%"
                line["E2E"] = f"{ev['summary']['e2e_mean_ms']:.2f}"
                line["Inf"] = f"{ev['summary']['inf_mean_ms']:.2f}"
            else:
                line["SR"] = line["E2E"] = line["Inf"] = "-"
        else:
            ev = j.get("eval", {})
            ep = "-"
            eta = "-"
            if j.get("status") == "running" and j.get("started"):
                d = P.RES / j["out"] / "episodes"
                n = sum(1 for _ in d.glob("*.json")) if d.exists() else 0
                elapsed = max(time.time() - j["started"], 1e-9)
                rate = n / elapsed
                ep = f"{n}/2498"
                eta = fmt_eta((2498 - n) / rate) if rate > 0 else "-"
            line["ep"] = ep
            line["ma100_e2e"] = line["best_dev"] = "-"
            line["eta"] = eta
            line["eval"] = ev.get("status", "-")
            if ev.get("summary"):
                line["SR"] = f"{ev['summary']['sr']*100:.2f}%"
                line["E2E"] = f"{ev['summary']['e2e_mean_ms']:.2f}"
                line["Inf"] = f"{ev['summary']['inf_mean_ms']:.2f}"
            else:
                line["SR"] = line["E2E"] = line["Inf"] = "-"
        rows.append(line)
    cols = ["id", "status", "ep", "ma100_e2e", "best_dev", "eta", "eval",
            "SR", "E2E", "Inf"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) if rows else len(c)
              for c in cols}
    out = [f"[{stage}] " + time.strftime("%Y-%m-%d %H:%M:%S")]
    out.append(" | ".join(c.ljust(widths[c]) for c in cols))
    out.append("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        out.append(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))
    return "\n".join(out)


def print_status(stage, status_path):
    st = load_status(status_path)
    print(table(st, stage), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["baselines", "ppo", "dqn"], required=True)
    ap.add_argument("--parallel-train", type=int, default=4)
    ap.add_argument("--parallel-eval", type=int, default=4)
    ap.add_argument("--parallel-baseline", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--report-every", type=int, default=60)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--wait-file", default="",
                    help="do not launch jobs until this file exists")
    a = ap.parse_args()

    SCHED_DIR.mkdir(parents=True, exist_ok=True)
    status_path = SCHED_DIR / f"{a.stage}_status.json"
    log_path = SCHED_DIR / f"{a.stage}.log"

    if a.status:
        print_status(a.stage, status_path)
        return 0
    if a.reset and status_path.exists():
        status_path.unlink()

    st = load_status(status_path)
    st["stage"] = a.stage
    st["started"] = st.get("started", time.time())
    jobs = build_jobs(a.stage)
    for j in jobs:
        if j["id"] not in st["jobs"]:
            st["jobs"][j["id"]] = {**j, "status": "pending", "pid": None,
                                   "progress": {}, "eval": {"status": "pending"}}
    save_status(status_path, st)

    env = dict(os.environ)
    env.setdefault("CITYNAV_DATA", "/home/20261024_1/Workspace/DATA")
    env.setdefault("STAGE12_TORCH_THREADS", "4")
    env.setdefault("OMP_NUM_THREADS", "4")
    procs = {}

    def launch(j):
        out_dir = P.RES / j["out"]
        out_dir.mkdir(parents=True, exist_ok=True)
        if j["kind"] == "baseline":
            log = open(SCHED_DIR / f"{j['id'].replace(':','_')}.log", "a")
            cmd = [PY, str(HERE / "run_association_closed_loop.py"),
                   "--group", j["group"], "--strategy", j["tag"],
                   "--seed", str(j["seed"]), "--out-dir", j["out"],
                   "--device", "cuda"]
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 env=env)
            j["pid"] = p.pid
            j["status"] = "running"
            j["started"] = time.time()
            procs[p.pid] = (j, p, log, None)
        else:
            log = open(SCHED_DIR / f"{j['id'].replace(':','_')}.log", "a")
            cmd = [PY, str(HERE / "train_association_rl.py"),
                   "--group", j["group"], "--algorithm", j["tag"],
                   "--seed", str(j["seed"]), "--episodes", str(a.episodes),
                   "--out-dir", j["out"], "--device", "cuda"]
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 env=env)
            j["pid"] = p.pid
            j["status"] = "running"
            j["started"] = time.time()
            procs[p.pid] = (j, p, log, None)
        save_status(status_path, st)

    def launch_eval(j):
        log = open(SCHED_DIR / f"{j['id'].replace(':','_')}_eval.log", "a")
        cmd = [PY, str(HERE / "eval_association_rl.py"),
               "--run-dir", j["out"],
               "--out-dir", j["eval_out"], "--device", "cuda"]
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        j["eval"] = {"status": "running", "pid": p.pid,
                     "started": time.time()}
        procs[p.pid] = (j, p, log, "eval")
        save_status(status_path, st)

    last_report = 0.0
    while True:
        if a.wait_file and not Path(a.wait_file).exists():
            if time.time() - last_report >= a.report_every:
                print(f"[{a.stage}] waiting for {a.wait_file}", flush=True)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f"[{a.stage}] waiting for {a.wait_file}\n")
                last_report = time.time()
            time.sleep(10)
            continue
        # launch pending jobs
        running_train = sum(1 for j in st["jobs"].values()
                            if j["status"] == "running")
        running_baseline = sum(1 for j in st["jobs"].values()
                               if j["kind"] == "baseline"
                               and j["status"] == "running")
        running_eval = sum(1 for j in st["jobs"].values()
                           if j.get("eval", {}).get("status") == "running")
        for j in st["jobs"].values():
            if j["status"] != "pending":
                continue
            if j["kind"] == "baseline":
                if running_baseline < a.parallel_baseline:
                    launch(j)
                    running_baseline += 1
            else:
                if running_train < a.parallel_train:
                    launch(j)
                    running_train += 1

        # update progress
        for j in st["jobs"].values():
            if j["status"] == "running" and j["kind"] == "train":
                j["progress"] = history_progress(P.RES / j["out"], a.episodes)
            if j.get("eval", {}).get("status") == "running":
                j["eval"]["progress"] = eval_progress(P.RES / j["eval_out"])

        # reap finished subprocesses
        finished = []
        for pid, (j, p, log, kind) in list(procs.items()):
            if p.poll() is not None:
                rc = p.returncode
                log.close()
                finished.append(pid)
                if kind is None:
                    j["status"] = "done" if rc == 0 else "failed"
                    j["returncode"] = rc
                    j["finished"] = time.time()
                    if rc != 0:
                        j["error"] = f"exit {rc}"
                    if j["kind"] == "baseline":
                        summ = P.RES / j["out"] / "summary.json"
                        if summ.exists():
                            try:
                                j["eval"] = {"status": "done",
                                             "summary": json.loads(
                                                 summ.read_text())}
                            except json.JSONDecodeError:
                                j["eval"] = {"status": "done"}
                        else:
                            j["eval"] = {"status": "failed",
                                         "error": "no summary"}
                    elif rc == 0:
                        launch_eval(j)
                    else:
                        j["eval"] = {"status": "failed",
                                     "error": "training failed"}
                else:
                    j["eval"]["status"] = ("done" if rc == 0 else "failed")
                    j["eval"]["returncode"] = rc
                    j["eval"]["finished"] = time.time()
                    summ = P.RES / j["eval_out"] / "summary.json"
                    if summ.exists():
                        try:
                            j["eval"]["summary"] = json.loads(
                                summ.read_text())
                        except json.JSONDecodeError:
                            pass
                    if rc != 0:
                        j["eval"]["error"] = f"exit {rc}"
                del procs[pid]
        save_status(status_path, st)

        if time.time() - last_report >= a.report_every:
            txt = table(st, a.stage)
            print(txt, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(txt + "\n")
            last_report = time.time()

        all_done = all(j["status"] in ("done", "failed")
                       and (j["kind"] == "baseline"
                            or j.get("eval", {}).get("status") in
                            ("done", "failed"))
                       for j in st["jobs"].values())
        if all_done and not procs:
            print(table(st, a.stage), flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(table(st, a.stage) + "\n")
            save_status(status_path, st)
            return 0
        time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
