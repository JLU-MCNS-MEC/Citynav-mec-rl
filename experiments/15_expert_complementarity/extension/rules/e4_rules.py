#!/usr/bin/env python3
"""Stage 15-Extension E4: do simple association rules beat a fixed expert?

Closed-loop evaluation on Train/Dev task pools (Val-Seen untouched) of:
  * fixed-expert policies      always_small / always_ts
  * static association         min_latency / max_quality / random
  * consensus routers          router_majority / router_conf

The routers need all five experts' first-step actions, so `RouterEnv` reuses
EpisodeEnv16's preprocessing on a *copy* of the belief (the real update and
the real step are still executed by EpisodeEnv16 itself -- the environment is
never modified).

Every step is also charged with the UAV energy model so the same run feeds the
F2 budget sweep.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
REPO = EXT.parents[2]
for p in [EXT, EXT / "energy", EXT / "rules",
          REPO / "experiments/16_cumulative_e2e_association/p1",
          REPO / "experiments/16_cumulative_e2e_association/p2",
          REPO / "experiments/12_hetero_mec_association/scripts"]:
    sys.path.insert(0, str(p))

from baselines import choose                       # noqa: E402
from energy_model import step_energy               # noqa: E402
from env16 import (MODEL_ORDER, NORMALIZED, EpisodeEnv16,  # noqa: E402
                   build_deployment_by_map, load_models, load_norm)
from env16 import (history_inputs, belief_features, landmark_inputs,  # noqa: E402
                   pool_history_from_current, EMPTY_BELIEF, EMPTY_MATCH,
                   matching, subpixel_match, AGREE_M, TAU, CAL, BEL, FORWARD_M)
from train_association_rl import load_tasks_and_split  # noqa: E402


def available_nodes(env):
    ms = env.metrics()
    idx = [i for i, m in enumerate(ms) if m.available]
    return idx if idx else list(range(len(ms))), ms


def nodes_of(env, mtype, ms=None):
    if ms is None:
        _, ms = available_nodes(env)
    return [i for i, m in enumerate(ms) if m.node["model_type"] == mtype
            and m.available]


def cheapest(env, mtype):
    idx, ms = available_nodes(env)
    cand = nodes_of(env, mtype, ms)
    if not cand:
        return min(idx, key=lambda i: ms[i].e2e_ms)
    return min(cand, key=lambda i: ms[i].e2e_ms)


class RouterEnv(EpisodeEnv16):
    """EpisodeEnv16 plus a read-only pass over all five experts."""

    def expert_actions(self):
        saved_belief = copy.deepcopy(self.belief)
        try:
            if self.empty_cands:
                belief_feat = EMPTY_BELIEF.copy()
                match_feat = EMPTY_MATCH.copy()
            else:
                obs = matching.prepare_observation(self.raster, self.pos[0],
                                                   self.pos[1], self.yaw)
                of = self.matcher_enc.features(obs[None], ("layer2",))["layer2"][0]
                mm = subpixel_match(of, self._gl2, 2)
                dx, dy = matching.pixel_to_offset(mm["col"], mm["row"])
                z = np.array([self.pos[0] + dx, self.pos[1] + dy])
                score, margin = mm["score"], mm["margin"]
                agree = (self.belief.last_accept is not None and
                         float(np.linalg.norm(z - self.belief.last_accept)) <= AGREE_M)
                if score >= TAU:
                    self.belief.last_accept = z.copy()
                if score >= TAU and agree:
                    self.belief.update(z, CAL["sigma_vis_m"], BEL["best"]["eta"],
                                       BEL["best"]["rho"], self.step_idx)
                est = self.belief.estimate()
                unc = float(math.sqrt(np.mean(self.belief.vars)))
                since = float(self.step_idx - self.belief.last_update)
                ent = float(self.belief.entropy())
                if not math.isfinite(ent):
                    ent = 0.0
                belief_feat = belief_features(est, self.pos, self.yaw, score, unc,
                                              ent, since, dx, dy)
                match_feat = np.asarray([dx, dy, score, margin, mm["sharpness"]],
                                        np.float32)
            vis, pose_h, act_h, hmask = history_inputs(
                self.hist[0], self.hist[1], self.hist[2], self.pos, self.yaw)
            cur_img = self.raster.heading_crop(float(self.pos[0]),
                                               float(self.pos[1]), float(self.yaw))
            out = {}
            with torch.no_grad():
                cur = self.enc4(np.asarray([cur_img], np.uint8))
                for mtype in MODEL_ORDER:
                    bf, mf, ph = belief_feat, match_feat, pose_h
                    if mtype in NORMALIZED:
                        bf = ((bf - self.norm["belief"][0]) /
                              self.norm["belief"][1]).astype(np.float32)
                        mf = ((mf - self.norm["match"][0]) /
                              self.norm["match"][1]).astype(np.float32)
                        ph = ((ph - self.norm["history_pose"][0]) /
                              self.norm["history_pose"][1]).astype(np.float32)
                    batch = {
                        "target_emb": torch.from_numpy(self.gdesc[None].copy()).to(self.device),
                        "history_vis": torch.from_numpy(vis).to(self.device),
                        "history_pose": torch.from_numpy(ph).to(self.device),
                        "history_action": torch.from_numpy(act_h).to(self.device),
                        "history_mask": torch.from_numpy(hmask).to(self.device),
                        "belief": torch.from_numpy(bf[None].copy()).to(self.device),
                        "match": torch.from_numpy(mf[None].copy()).to(self.device),
                        "landmarks": torch.from_numpy(
                            landmark_inputs(self.belief, self.empty_cands)).to(self.device),
                        "states": torch.from_numpy(
                            np.asarray([[self.pos[0], self.pos[1], self.yaw]],
                                       np.float32)).to(self.device),
                        "current": cur,
                    }
                    logits = self.models[mtype](batch).float()[0]
                    probs = torch.softmax(logits, -1)
                    out[mtype] = {"action": int(probs.argmax()),
                                  "conf": float(probs.max())}
        finally:
            self.belief = saved_belief
        return out


def pick_static(env, policy, rng):
    idx, ms = available_nodes(env)
    if policy == "random":
        return int(rng.choice(idx))
    if policy == "min_latency":
        return min(idx, key=lambda i: ms[i].e2e_ms)
    if policy == "max_quality":
        return cheapest(env, "ts")
    if policy.startswith("always_"):
        return cheapest(env, policy.split("_", 1)[1])
    raise ValueError(policy)


def pick_router(env, policy, acts):
    """Both rules fall back to `small` when every expert agrees: the executed
    action is then identical for all five, so the cheapest model costs the
    least latency without changing the trajectory."""
    from collections import Counter
    if len({a["action"] for a in acts.values()}) == 1:
        return "small"
    if policy == "router_majority":
        cnt = Counter(a["action"] for a in acts.values())
        top = max(cnt.values())
        tied = [k for k, a in acts.items() if cnt[a["action"]] == top]
        return min(tied, key=lambda m: MODEL_ORDER.index(m))
    if policy == "router_conf":
        return max(acts, key=lambda m: acts[m]["conf"])
    raise ValueError(policy)


def run(env, tasks, policy, rng):
    rows = []
    for t in tasks:
        env.start(t)
        done = False
        energies, e2es, acts_hist = [], [], []
        while not done:
            if policy.startswith("router_"):
                acts = env.expert_actions()
                mtype = pick_router(env, policy, acts)
                node = cheapest(env, mtype)
            else:
                node = pick_static(env, policy, rng)
            _, done, info = env.step(node)
            last = env.steps[-1]
            a = int(last["action"])
            energies.append(step_energy(a, env.enc_ms / 1e3,
                                        last["comm_ms"] / 1e3,
                                        last["inf_ms"] / 1e3)["e_total_j"])
            e2es.append(float(last["e2e_ms"]))
            acts_hist.append(a)
        rows.append({
            "episode_id": t.get("episode_id", ""), "map": t["map"],
            "split": t.get("split", ""), "policy": policy,
            "success": bool(env.success), "failure": env.failure,
            "steps": env.step_idx, "cum_e2e_ms": float(np.sum(env.episode_e2e)),
            "energy_j": float(np.sum(energies)),
            "n_forward": env.n_forward,
            "stop_actions": int(sum(1 for a in acts_hist if a == 0)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=150)
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--policies", default="always_small,always_ts,min_latency,"
                                          "max_quality,random,router_majority,router_conf")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(HERE / "e4_rules_results.csv"))
    args = ap.parse_args()
    torch.set_num_threads(4)

    train_tasks, dev_tasks = load_tasks_and_split()
    sel = np.random.default_rng(args.seed)
    tr = [train_tasks[i] for i in sel.permutation(len(train_tasks))[: args.n_train]]
    dv = [dev_tasks[i] for i in sel.permutation(len(dev_tasks))[: args.n_dev]]
    for t in tr:
        t["split"] = "train"
    for t in dv:
        t["split"] = "dev"
    tasks = tr + dv

    env = RouterEnv(build_deployment_by_map(), args.device,
                    load_models(args.device), load_norm(),
                    reward_variant="A", record_steps=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    for pol in args.policies.split(","):
        new = run(env, tasks, pol, rng)
        rows += new
        df = pd.DataFrame(new)
        d = df[df.split == "dev"]
        print(f"[{pol}] dev SR={d.success.mean():.3f} "
              f"E2E={d.cum_e2e_ms.mean():.0f}ms E={d.energy_j.mean():.0f}J "
              f"steps={d.steps.mean():.1f}", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    summ = {}
    for (pol, split), g in df.groupby(["policy", "split"]):
        summ[f"{pol}/{split}"] = {
            "n": int(len(g)), "sr": float(g.success.mean()),
            "e2e_ms_mean": float(g.cum_e2e_ms.mean()),
            "e2e_ms_median": float(g.cum_e2e_ms.median()),
            "energy_j_mean": float(g.energy_j.mean()),
            "energy_j_median": float(g.energy_j.median()),
            "steps_mean": float(g.steps.mean()),
            "oob": int((g.failure == "out_of_bounds").sum()),
            "premature": int((g.failure == "premature_stop").sum()),
            "collision": int((g.failure == "collision").sum()),
            "step_cap": int((g.failure == "step_cap").sum()),
        }
    Path(args.out).with_suffix(".json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
