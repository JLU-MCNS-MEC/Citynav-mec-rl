#!/usr/bin/env python3
"""Stage 15-Extension G: PPO on the dual-constraint CMDP (functional test).

Lagrangian formulation
    r'_t = r_t - lambda_E * (e_t / B) - [terminal and failure] * lambda_SR
    lambda_E  += eta * (EMA[E/B] - 1)              clipped to [0, LAM_MAX]
    lambda_SR += eta * (EMA[fail] - (1 - SR_TARGET))

The reward is the Stage-16 variant A (-e2e/100); energy and success enter only
as constraints, never as reward shaping.  With --fixed-lambdas 0 0 the script
reproduces the unconstrained PPO on the same task subset (ablation).

Runs are short on purpose: 300 episodes is a functional check, not a
convergence study.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
REPO = EXT.parents[2]
for p in [HERE, EXT / "energy", REPO / "experiments/16_cumulative_e2e_association/p1",
          REPO / "experiments/16_cumulative_e2e_association/p3",
          REPO / "experiments/12_hetero_mec_association/scripts"]:
    sys.path.insert(0, str(p))

from env_constrained import ConstrainedEnv, load_budgets     # noqa: E402
from env16 import build_deployment_by_map, load_models, load_norm  # noqa: E402
from train16 import (N_ACT, NORM_MS, STATE_DIM, ActorCritic,   # noqa: E402
                     masked_softmax, state_features)
from train_association_rl import load_tasks_and_split          # noqa: E402

LAM_MAX = 50.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budget-file", default=str(HERE / "budget_reference.json"))
    ap.add_argument("--slack", type=float, default=1.2)
    ap.add_argument("--sr-target", type=float, default=0.5)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--ema", type=int, default=20)
    ap.add_argument("--fixed-lambdas", nargs=2, type=float, default=None,
                    metavar=("LAM_E", "LAM_SR"))
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--n-train-pool", type=int, default=0,
                    help="cap on budgeted Train tasks (0 = all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CMDP_THREADS", "4")))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = args.device

    budgets = load_budgets(args.budget_file)
    train_tasks, dev_tasks = load_tasks_and_split()
    tr = [t for t in train_tasks if int(t["index"]) in budgets]
    if args.n_train_pool:
        tr = tr[: args.n_train_pool]
    dv = [t for t in dev_tasks if int(t["index"]) in budgets]
    print(f"train tasks with budget: {len(tr)}  dev: {len(dv)}", flush=True)

    env = ConstrainedEnv(build_deployment_by_map(), device,
                         load_models(device), load_norm(),
                         reward_variant="A", norm_ms=NORM_MS,
                         budgets=budgets, slack=args.slack,
                         v_forward=float(json.loads(
                             Path(args.budget_file).read_text())["v_forward"]))

    ac = ActorCritic().to(device)
    opt = torch.optim.Adam(ac.parameters(), lr=3e-4)
    gamma = 0.99

    if args.fixed_lambdas:
        lam_e, lam_sr = float(args.fixed_lambdas[0]), float(args.fixed_lambdas[1])
        learn_lambda = False
    else:
        lam_e, lam_sr = 0.0, 0.0
        learn_lambda = True
    ema_ratio = deque(maxlen=args.ema)
    ema_fail = deque(maxlen=args.ema)

    hist = out / "history.jsonl"
    devp = out / "dev.jsonl"
    t0 = time.time()
    buf = {k: [] for k in ("s", "m", "a", "r", "d", "lp", "v")}

    def run_episode(task, training=True):
        env.start(task)
        s, mask = state_features(env)
        done = False
        ep_r = 0.0
        while not done:
            st = torch.as_tensor(s[None], device=device)
            with torch.no_grad():
                logits, val = ac(st)
                probs = masked_softmax(logits, torch.as_tensor(mask[None], device=device))
            a = int(torch.multinomial(probs[0], 1).item())
            lp = float(torch.log(probs[0, a] + 1e-12))
            base_r, done, info = env.step(a)
            r = base_r
            if env.has_budget:
                r -= lam_e * info["step_energy_j"] / max(env.budget_j, 1e-9)
            if done and not env.success:
                r -= lam_sr
            s2, m2 = state_features(env)
            if training:
                buf["s"].append(s); buf["m"].append(mask); buf["a"].append(a)
                buf["r"].append(r); buf["d"].append(float(done))
                buf["lp"].append(lp); buf["v"].append(float(val[0]))
            ep_r += r
            s, mask = s2, m2
        return ep_r

    def ppo_update():
        S = torch.as_tensor(np.asarray(buf["s"]), device=device)
        M = torch.as_tensor(np.asarray(buf["m"]), device=device)
        A = torch.as_tensor(np.asarray(buf["a"]), device=device)
        R = torch.as_tensor(np.asarray(buf["r"]), dtype=torch.float32, device=device)
        D = torch.as_tensor(np.asarray(buf["d"]), dtype=torch.float32, device=device)
        LP = torch.as_tensor(np.asarray(buf["lp"]), dtype=torch.float32, device=device)
        V = torch.as_tensor(np.asarray(buf["v"]), dtype=torch.float32, device=device)
        adv = torch.zeros_like(R); gae = 0.0
        for t in range(len(R) - 1, -1, -1):
            nv = V[t + 1] if t + 1 < len(V) else 0.0
            delta = R[t] + gamma * (1 - D[t]) * nv - V[t]
            gae = delta + gamma * 0.95 * (1 - D[t]) * gae
            adv[t] = gae
        ret = adv + V
        if len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        for _ in range(4):
            logits, val = ac(S)
            p = masked_softmax(logits, M)
            logp = torch.log(p.gather(1, A[:, None]).squeeze(1) + 1e-12)
            ratio = torch.exp(logp - LP)
            pi_loss = -torch.min(ratio * adv,
                                 torch.clamp(ratio, 0.8, 1.2) * adv).mean()
            loss = pi_loss + 0.5 * (val - ret).pow(2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
            opt.step()
        for k in buf:
            buf[k] = []

    def evaluate():
        rows = []
        for task in dv:
            run_episode(task, training=False)
            rows.append({"index": int(task["index"]), "success": bool(env.success),
                         "failure": env.failure, "steps": env.step_idx,
                         "energy_j": env.energy_j, "budget_j": env.budget_j,
                         "ratio": env.energy_ratio(),
                         "cum_e2e_ms": float(np.sum(env.episode_e2e))})
        return {"n": len(rows), "sr": float(np.mean([r["success"] for r in rows])),
                "viol_rate": float(np.mean([r["ratio"] > 1 for r in rows])),
                "mean_ratio": float(np.mean([r["ratio"] for r in rows])),
                "mean_energy_j": float(np.mean([r["energy_j"] for r in rows])),
                "mean_cum_e2e_ms": float(np.mean([r["cum_e2e_ms"] for r in rows])),
                "premature": int(sum(r["failure"] == "premature_stop" for r in rows)),
                "oob": int(sum(r["failure"] == "out_of_bounds" for r in rows)),
                "rows": rows}

    for ep in range(1, args.episodes + 1):
        task = tr[(ep - 1) % len(tr)]
        ep_r = run_episode(task)
        ppo_update()
        c = env.costs()
        ema_ratio.append(c["energy_ratio"]); ema_fail.append(c["fail"])
        if learn_lambda and len(ema_ratio) >= 5:
            lam_e = float(np.clip(lam_e + args.eta * (np.mean(ema_ratio) - 1.0),
                                  0.0, LAM_MAX))
            lam_sr = float(np.clip(lam_sr + args.eta * (np.mean(ema_fail)
                                                        - (1 - args.sr_target)),
                                   0.0, LAM_MAX))
        rec = {"episode": ep, "reward": ep_r, "base_success": bool(env.success),
               "failure": env.failure, "steps": env.step_idx,
               "energy_j": env.energy_j, "budget_j": env.budget_j,
               "energy_ratio": c["energy_ratio"],
               "cum_e2e_ms": float(np.sum(env.episode_e2e)),
               "lam_e": lam_e, "lam_sr": lam_sr,
               "ema_ratio": float(np.mean(ema_ratio)),
               "ema_fail": float(np.mean(ema_fail)),
               "elapsed_sec": round(time.time() - t0, 1)}
        with hist.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if ep % 50 == 0:
            print(f"ep {ep} SR(1ep)={rec['base_success']} E/B={c['energy_ratio']:.2f} "
                  f"lamE={lam_e:.2f} lamSR={lam_sr:.2f}", flush=True)
        if args.eval_every and ep % args.eval_every == 0:
            d = evaluate()
            with devp.open("a", encoding="utf-8") as f:
                f.write(json.dumps(d) + "\n")
            print(f"  [dev] SR={d['sr']:.3f} viol={d['viol_rate']:.2f} "
                  f"E/B={d['mean_ratio']:.2f} E={d['mean_energy_j']:.1f}J "
                  f"cumE2E={d['mean_cum_e2e_ms']:.0f}ms prem={d['premature']} "
                  f"oob={d['oob']}", flush=True)
            torch.save({"ep": ep, "state_dim": STATE_DIM, "n_act": N_ACT,
                        "policy": {"ac": ac.state_dict()},
                        "lam_e": lam_e, "lam_sr": lam_sr},
                       out / "checkpoint.pt")

    torch.save({"ep": args.episodes, "state_dim": STATE_DIM, "n_act": N_ACT,
                "policy": {"ac": ac.state_dict()}, "lam_e": lam_e,
                "lam_sr": lam_sr, "slack": args.slack,
                "sr_target": args.sr_target},
               out / "final.pt")
    print(json.dumps({"episodes": args.episodes, "lam_e": lam_e,
                      "lam_sr": lam_sr, "elapsed_sec": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
