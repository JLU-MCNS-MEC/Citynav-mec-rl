#!/usr/bin/env python3
"""Stage 15-Extension G4: DQN / TD3 / DDPG on the dual-constraint CMDP.

Same environment, reward shaping and Lagrangian dual ascent as
train_cmdp_ppo.py / train_cmdp_sac.py, same logging schema, so all five
algorithms share the MA(100) plotting and evaluation tooling.

The learners are the Stage-16 implementations unchanged:
  dqn : Double-DQN + target net + replay
  td3 : discrete TD3 (twin Q, policy delay, target smoothing by sampling the
        target policy's softmax)
  ddpg: discrete DDPG (twin-free Q, deterministic argmax target policy)
Only the reward carries the Lagrangian terms.
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

from env_constrained import ConstrainedEnv, load_budgets      # noqa: E402
from env16 import build_deployment_by_map, load_models, load_norm  # noqa: E402
from train16 import (NORM_MS, QNet, ActorCritic, Replay,      # noqa: E402
                     masked_softmax, state_features)
from train_association_rl import load_tasks_and_split         # noqa: E402

LAM_MAX = 50.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", choices=["dqn", "td3", "ddpg"], required=True)
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
    print(f"[{args.algorithm}] train tasks: {len(tr)}  dev: {len(dv)}", flush=True)

    env = ConstrainedEnv(build_deployment_by_map(), device,
                         load_models(device), load_norm(),
                         reward_variant="A", norm_ms=NORM_MS,
                         budgets=budgets, slack=args.slack,
                         v_forward=float(json.loads(
                             Path(args.budget_file).read_text())["v_forward"]))

    algo = args.algorithm
    replay = Replay(seed=args.seed)
    if algo == "dqn":
        q, qt = QNet().to(device), QNet().to(device)
        qt.load_state_dict(q.state_dict())
        opt = torch.optim.Adam(q.parameters(), lr=1e-4)
    elif algo == "td3":
        q1, q2 = QNet().to(device), QNet().to(device)
        q1t, q2t = QNet().to(device), QNet().to(device)
        q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
        pi, pit = ActorCritic().to(device), ActorCritic().to(device)
        pit.load_state_dict(pi.state_dict())
        opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)
        opt_pi = torch.optim.Adam(pi.parameters(), lr=3e-4)
    else:  # ddpg
        q, qt = QNet().to(device), QNet().to(device)
        qt.load_state_dict(q.state_dict())
        pi, pit = ActorCritic().to(device), ActorCritic().to(device)
        pit.load_state_dict(pi.state_dict())
        opt_q = torch.optim.Adam(q.parameters(), lr=3e-4)
        opt_pi = torch.optim.Adam(pi.parameters(), lr=3e-4)

    def policy_logits(s):
        with torch.no_grad():
            if algo == "dqn":
                return q(torch.as_tensor(s[None], device=device))[0].cpu().numpy()
            if algo == "td3":
                lg, _ = pi(torch.as_tensor(s[None], device=device))
            else:
                lg, _ = pi(torch.as_tensor(s[None], device=device))
            return lg[0].cpu().numpy()

    def greedy(s, mask):
        qv = policy_logits(s)
        qv = qv.copy(); qv[np.asarray(mask) < 0.5] = -1e18
        return int(np.argmax(qv))

    def policy_state():
        if algo == "dqn":
            return {"q": q.state_dict()}
        if algo == "td3":
            return {"q1": q1.state_dict(), "q2": q2.state_dict(),
                    "pi": pi.state_dict()}
        return {"q": q.state_dict(), "pi": pi.state_dict()}

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

    lam_e, lam_sr = 0.0, 0.0
    ema_ratio, ema_fail = deque(maxlen=args.ema), deque(maxlen=args.ema)
    hist, devp = out / "history.jsonl", out / "dev.jsonl"
    t0 = time.time(); total_steps = 0

    for ep in range(1, args.episodes + 1):
        env.start(tr[(ep - 1) % len(tr)])
        s, mask = state_features(env)
        done = False; ep_r = 0.0
        while not done:
            eps = max(0.05, 1.0 - total_steps / 60000)
            if random.random() < eps:
                a = int(np.random.choice(np.flatnonzero(np.asarray(mask) > 0.5)))
            else:
                a = greedy(s, mask)
            base_r, done, info = env.step(a)
            r = base_r
            if env.has_budget:
                r -= lam_e * info["step_energy_j"] / max(env.budget_j, 1e-9)
            if done and not env.success:
                r -= lam_sr
            s2, m2 = state_features(env)
            replay.add(s, mask, a, r, s2, m2, float(done))
            ep_r += r; total_steps += 1

            if replay.size >= 2000:
                bs, bm, ba, br, bs2, bm2, bd = replay.sample(128, device)
                if algo == "dqn":
                    with torch.no_grad():
                        qn = qt(bs2).masked_fill(bm2 < 0.5, -1e18).max(1).values
                        tgt = br + args.gamma * (1 - bd) * qn
                    qv = q(bs).gather(1, ba[:, None]).squeeze(1)
                    loss = F.smooth_l1_loss(qv, tgt)
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), 1.0); opt.step()
                    if total_steps % 200 == 0:
                        qt.load_state_dict(q.state_dict())
                elif algo == "td3":
                    with torch.no_grad():
                        lg2, _ = pit(bs2)
                        p2 = masked_softmax(lg2, bm2)
                        a2 = torch.multinomial(p2, 1).squeeze(1)
                        qv = torch.min(q1t(bs2), q2t(bs2)).gather(
                            1, a2[:, None]).squeeze(1)
                        tgt = br + args.gamma * (1 - bd) * qv
                    qv1 = q1(bs).gather(1, ba[:, None]).squeeze(1)
                    qv2 = q2(bs).gather(1, ba[:, None]).squeeze(1)
                    loss_q = F.smooth_l1_loss(qv1, tgt) + F.smooth_l1_loss(qv2, tgt)
                    opt_q.zero_grad(); loss_q.backward()
                    nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 1.0)
                    opt_q.step()
                    if total_steps % 200 == 0:
                        lg, _ = pi(bs)
                        p = masked_softmax(lg, bm)
                        a_pi = torch.multinomial(p.detach(), 1).squeeze(1)
                        loss_pi = -torch.min(q1(bs), q2(bs)).gather(
                            1, a_pi[:, None]).mean()
                        opt_pi.zero_grad(); loss_pi.backward()
                        nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt_pi.step()
                        for t_, o_ in zip(pit.parameters(), pi.parameters()):
                            t_.data.mul_(1 - args.tau).add_(args.tau * o_.data)
                        for t_, o_ in zip(q1t.parameters(), q1.parameters()):
                            t_.data.mul_(1 - args.tau).add_(args.tau * o_.data)
                        for t_, o_ in zip(q2t.parameters(), q2.parameters()):
                            t_.data.mul_(1 - args.tau).add_(args.tau * o_.data)
                else:  # ddpg
                    with torch.no_grad():
                        lg2, _ = pit(bs2)
                        p2 = masked_softmax(lg2, bm2)
                        a2 = p2.argmax(1)
                        qn = qt(bs2).gather(1, a2[:, None]).squeeze(1)
                        tgt = br + args.gamma * (1 - bd) * qn
                    qv = q(bs).gather(1, ba[:, None]).squeeze(1)
                    loss_q = F.smooth_l1_loss(qv, tgt)
                    opt_q.zero_grad(); loss_q.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), 1.0); opt_q.step()
                    lg, _ = pi(bs)
                    p = masked_softmax(lg, bm)
                    loss_pi = -(p * q(bs)).sum(1).mean()
                    opt_pi.zero_grad(); loss_pi.backward()
                    nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt_pi.step()
                    for t_, o_ in zip(qt.parameters(), q.parameters()):
                        t_.data.mul_(1 - args.tau).add_(args.tau * o_.data)
                    for t_, o_ in zip(pit.parameters(), pi.parameters()):
                        t_.data.mul_(1 - args.tau).add_(args.tau * o_.data)
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
            print(f"[{algo}] ep {ep} SR(1ep)={rec['base_success']} "
                  f"E/B={c['energy_ratio']:.2f} lamE={lam_e:.2f} lamSR={lam_sr:.2f}",
                  flush=True)
        if args.eval_every and ep % args.eval_every == 0:
            d = evaluate()
            with devp.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"episode": ep, **d}) + "\n")
            print(f"  [dev] SR={d['sr']:.3f} viol={d['viol_rate']:.2f} "
                  f"E/B={d['mean_ratio']:.2f} E={d['mean_energy_j']:.1f}J "
                  f"E2E={d['mean_cum_e2e_ms']:.0f}ms prem={d['premature']} oob={d['oob']}",
                  flush=True)
            torch.save({"ep": ep, "algorithm": algo, "policy": policy_state(),
                        "lam_e": lam_e, "lam_sr": lam_sr}, out / "checkpoint.pt")

    torch.save({"ep": args.episodes, "algorithm": algo, "policy": policy_state(),
                "lam_e": lam_e, "lam_sr": lam_sr, "slack": args.slack,
                "sr_target": args.sr_target}, out / "final.pt")
    print(json.dumps({"algorithm": algo, "episodes": args.episodes,
                      "lam_e": lam_e, "lam_sr": lam_sr,
                      "elapsed_sec": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
