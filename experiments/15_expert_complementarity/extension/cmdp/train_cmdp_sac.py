#!/usr/bin/env python3
"""Stage 15-Extension G3: discrete SAC on the dual-constraint CMDP.

Same Lagrangian formulation and logging schema as train_cmdp_ppo.py, so the
two algorithms share the plotting / evaluation code:

    r'_t = r_t - lambda_E * (e_t / B) - [terminal and failure] * lambda_SR
    lambda_E  += eta * (EMA[E/B] - 1)
    lambda_SR += eta * (EMA[fail] - (1 - SR_TARGET))

The SAC learner itself is the Stage-16 discrete SAC (twin soft-Q, entropy
temperature auto-tuning, target networks); nothing in the algorithm changed.
"""
from __future__ import annotations

import argparse
import json
import math
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

from env_constrained import ConstrainedEnv, load_budgets      # noqa: E402
from env16 import build_deployment_by_map, load_models, load_norm  # noqa: E402
from train16 import (N_ACT, NORM_MS, QNet, ActorCritic, Replay,  # noqa: E402
                     masked_softmax, state_features)
from train_association_rl import load_tasks_and_split          # noqa: E402

LAM_MAX = 50.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budget-file", default=str(HERE / "budget_reference_1000.json"))
    ap.add_argument("--slack", type=float, default=0.8)
    ap.add_argument("--sr-target", type=float, default=0.5)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--ema", type=int, default=20)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--n-train-pool", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CMDP_THREADS", "4")))
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
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

    q1, q2 = QNet().to(device), QNet().to(device)
    q1t, q2t = QNet().to(device), QNet().to(device)
    q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
    ac = ActorCritic().to(device)
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)
    opt_pi = torch.optim.Adam(ac.parameters(), lr=3e-4)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    opt_alpha = torch.optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -0.5 * math.log(N_ACT)
    replay = Replay(seed=args.seed)

    lam_e, lam_sr = 0.0, 0.0
    ema_ratio, ema_fail = deque(maxlen=args.ema), deque(maxlen=args.ema)

    hist, devp = out / "history.jsonl", out / "dev.jsonl"
    t0 = time.time()
    total_steps = 0

    def act(s, mask):
        with torch.no_grad():
            logits, _ = ac(torch.as_tensor(s[None], device=device))
            probs = masked_softmax(logits, torch.as_tensor(mask[None], device=device))
        return int(torch.multinomial(probs[0], 1).item())

    def greedy(s, mask):
        with torch.no_grad():
            logits, _ = ac(torch.as_tensor(s[None], device=device))
            p = masked_softmax(logits, torch.as_tensor(mask[None], device=device))
        return int(p[0].argmax())

    def learn(bs, bm, ba, br, bs2, bm2, bd):
        with torch.no_grad():
            lg2, _ = ac(bs2)
            p2 = masked_softmax(lg2, bm2)
            logp2 = torch.log(p2 + 1e-12)
            alpha = log_alpha.exp()
            qn = torch.min(q1t(bs2), q2t(bs2)) - alpha * logp2
            qn = qn.masked_fill(bm2 < 0.5, -1e18).max(1).values
            tgt = br + args.gamma * (1 - bd) * qn
        qv1 = q1(bs).gather(1, ba[:, None]).squeeze(1)
        qv2 = q2(bs).gather(1, ba[:, None]).squeeze(1)
        loss_q = F.mse_loss(qv1, tgt) + F.mse_loss(qv2, tgt)
        opt_q.zero_grad(); loss_q.backward()
        nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 1.0)
        opt_q.step()
        lg, _ = ac(bs)
        p = masked_softmax(lg, bm)
        logp = torch.log(p + 1e-12)
        qpi = torch.min(q1(bs), q2(bs))
        loss_pi = (p * (log_alpha.exp() * logp - qpi)).sum(1).mean()
        opt_pi.zero_grad(); loss_pi.backward()
        nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
        opt_pi.step()
        loss_a = -(p.detach() * (log_alpha.exp() * logp.detach()
                                 + target_entropy)).sum(1).mean()
        opt_alpha.zero_grad(); loss_a.backward(); opt_alpha.step()
        for t_, o_ in zip(q1t.parameters(), q1.parameters()):
            t_.data.mul_(1 - args.tau).add_(args.tau * o_.data)
        for t_, o_ in zip(q2t.parameters(), q2.parameters()):
            t_.data.mul_(1 - args.tau).add_(args.tau * o_.data)

    def evaluate():
        rows = []
        for task in dv:
            env.start(task)
            s, mask = state_features(env)
            done = False
            while not done:
                a = greedy(s, mask)
                _, done, _ = env.step(a)
                s, mask = state_features(env)
            rows.append({"success": bool(env.success), "failure": env.failure,
                         "steps": env.step_idx, "energy_j": env.energy_j,
                         "budget_j": env.budget_j, "ratio": env.energy_ratio(),
                         "cum_e2e_ms": float(np.sum(env.episode_e2e))})
        return {"n": len(rows), "sr": float(np.mean([r["success"] for r in rows])),
                "viol_rate": float(np.mean([r["ratio"] > 1 for r in rows])),
                "mean_ratio": float(np.mean([r["ratio"] for r in rows])),
                "mean_energy_j": float(np.mean([r["energy_j"] for r in rows])),
                "mean_cum_e2e_ms": float(np.mean([r["cum_e2e_ms"] for r in rows])),
                "premature": int(sum(r["failure"] == "premature_stop" for r in rows)),
                "oob": int(sum(r["failure"] == "out_of_bounds" for r in rows))}

    for ep in range(1, args.episodes + 1):
        task = tr[(ep - 1) % len(tr)]
        env.start(task)
        s, mask = state_features(env)
        done = False
        ep_r = 0.0
        while not done:
            a = act(s, mask)
            base_r, done, info = env.step(a)
            r = base_r
            if env.has_budget:
                r -= lam_e * info["step_energy_j"] / max(env.budget_j, 1e-9)
            if done and not env.success:
                r -= lam_sr
            s2, m2 = state_features(env)
            replay.add(s, mask, a, r, s2, m2, float(done))
            if replay.size >= 2000:
                bs, bm, ba, br, bs2, bm2, bd = replay.sample(128, device)
                learn(bs, bm, ba, br, bs2, bm2, bd)
            ep_r += r
            total_steps += 1
            s, mask = s2, m2

        c = env.costs()
        ema_ratio.append(c["energy_ratio"]); ema_fail.append(c["fail"])
        if len(ema_ratio) >= 5:
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
               "mean_e2e_ms": float(np.mean(env.episode_e2e)) if env.episode_e2e else 0.0,
               "lam_e": lam_e, "lam_sr": lam_sr,
               "ema_ratio": float(np.mean(ema_ratio)),
               "ema_fail": float(np.mean(ema_fail)),
               "total_steps": total_steps,
               "elapsed_sec": round(time.time() - t0, 1)}
        with hist.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if ep % 100 == 0:
            print(f"ep {ep} SR(1ep)={rec['base_success']} E/B={c['energy_ratio']:.2f} "
                  f"lamE={lam_e:.2f} lamSR={lam_sr:.2f}", flush=True)
        if args.eval_every and ep % args.eval_every == 0:
            d = evaluate()
            with devp.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"episode": ep, **d}) + "\n")
            print(f"  [dev] SR={d['sr']:.3f} viol={d['viol_rate']:.2f} "
                  f"E/B={d['mean_ratio']:.2f} E={d['mean_energy_j']:.1f}J "
                  f"E2E={d['mean_cum_e2e_ms']:.0f}ms prem={d['premature']} oob={d['oob']}",
                  flush=True)
            torch.save({"ep": ep, "algorithm": "sac", "state_dim": None,
                        "policy": {"q1": q1.state_dict(), "q2": q2.state_dict(),
                                   "ac": ac.state_dict(),
                                   "log_alpha": log_alpha.detach().cpu()},
                        "lam_e": lam_e, "lam_sr": lam_sr}, out / "checkpoint.pt")

    torch.save({"ep": args.episodes, "algorithm": "sac",
                "policy": {"q1": q1.state_dict(), "q2": q2.state_dict(),
                           "ac": ac.state_dict(),
                           "log_alpha": log_alpha.detach().cpu()},
                "lam_e": lam_e, "lam_sr": lam_sr, "slack": args.slack,
                "sr_target": args.sr_target}, out / "final.pt")
    print(json.dumps({"episodes": args.episodes, "lam_e": lam_e, "lam_sr": lam_sr,
                      "elapsed_sec": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
