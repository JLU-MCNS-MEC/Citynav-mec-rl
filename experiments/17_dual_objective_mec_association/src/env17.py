#!/usr/bin/env python3
"""Stage 17-A dual-objective constrained MEC association environment.

Subclasses Stage 16's `EpisodeEnv16` and NEVER modifies it.  Navigation,
collision, stop detection and the frozen experts are all executed by the
parent; this class only adds:

  * a four-component collaborative service latency (enc + up + inf + down),
  * a per-step UAV energy model (service + motion),
  * the task-level energy budget and the per-service deadline constraints,
  * feasibility masks and step/episode records.

Constraint handling (spec sections VI and VII):

  1. the agent may only select a node whose `feasible_mec_mask` is 1;
  2. if no node is feasible the episode terminates with `no_feasible_mec`;
  3. the navigation action is previewed from a *copy* of the belief, so the
     motion energy can be checked BEFORE the UAV is allowed to move;
  4. if the motion energy does not fit the remaining budget the episode
     terminates with `energy_budget_exhausted` and the action is not executed.

Causality: the preview uses exactly the same observable inputs the expert
would receive at this step; nothing from the future is used.
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in [HERE, REPO / "experiments/16_cumulative_e2e_association/p1",
          REPO / "experiments/12_hetero_mec_association/scripts"]:
    sys.path.insert(0, str(p))

from env16 import (AGREE_M, BEL, CAL, EMPTY_BELIEF, EMPTY_MATCH, TAU,  # noqa: E402
                   EpisodeEnv16, belief_features, history_inputs,
                   landmark_inputs, matching, subpixel_match)
from config17 import Stage17Config                     # noqa: E402
from energy17 import e_coll_j, e_motion_j              # noqa: E402
from latency17 import service_latency_ms               # noqa: E402

NORMALIZED = {"ts", "tm", "large"}


class EpisodeEnv17(EpisodeEnv16):
    def __init__(self, by_map, device, models, norm, cfg: Stage17Config, **kw):
        kw.setdefault("record_steps", True)
        super().__init__(by_map, device, models, norm, reward_variant="A", **kw)
        self.cfg = cfg
        self.preview_mismatch_count = 0

    # ------------------------------------------------------------------
    # episode bookkeeping
    # ------------------------------------------------------------------
    def start(self, task):
        super().start(task)
        c = self.cfg
        # Episode budget = min(usable pack energy, task budget).  With the
        # review decision of 2026-11-26 the only energy constraint is
        # "episode energy does not exceed the battery capacity", so both
        # quantities default to 360,000 J (100 Wh) and the min() keeps the
        # two knobs from silently disagreeing.
        self.energy_budget_j = min(c.initial_energy_j - c.battery_safety_margin_j,
                                   c.task_energy_budget_j)
        self.remaining_energy_j = self.energy_budget_j
        self.cumulative_energy_j = 0.0
        self.cumulative_coll_latency_ms = 0.0
        self.energy_budget_violation_count = 0
        self.deadline_violation_count = 0
        self.invalid_mec_action_count = 0
        self.terminated_reason = ""
        self.constraint_failure = ""
        self.step_records = []
        self.model_selection = np.zeros(10, dtype=np.int64)

    # ------------------------------------------------------------------
    # candidate table / masks
    # ------------------------------------------------------------------
    def candidate_table(self) -> list[dict]:
        c = self.cfg
        rows = []
        for i, m in enumerate(self.metrics()):
            lat = service_latency_ms(m.distance, m.node["model_type"], c)
            cost = e_coll_j(lat, c)["e_coll_j"]
            cov = bool(m.available) if c.enforce_coverage else True
            dead = (lat["d_coll_ms"] <= c.d_max_ms) if c.enforce_deadline else True
            energy = (cost <= self.remaining_energy_j) if c.enforce_energy else True
            rows.append({
                "node": i, "gbs_id": m.node["gbs_id"],
                "model_type": m.node["model_type"],
                "distance_m": float(m.distance),
                "coverage_valid": cov, "deadline_valid": dead,
                "energy_valid": energy,
                "feasible": bool(cov and dead and energy),
                "latency": lat, "service_energy_j": cost,
            })
        return rows

    def masks(self) -> dict:
        table = self.candidate_table()
        arr = lambda k: np.asarray([r[k] for r in table], dtype=np.float32)  # noqa: E731
        return {
            "coverage_valid_mask": arr("coverage_valid"),
            "deadline_valid_mask": arr("deadline_valid"),
            "feasible_mec_mask": arr("feasible"),
            "candidate_service_latency_ms": np.asarray(
                [r["latency"]["d_coll_ms"] for r in table], np.float32),
            "candidate_energy_cost_j": np.asarray(
                [r["service_energy_j"] for r in table], np.float32),
        }

    def feasible_node_indices(self) -> list[int]:
        return [i for i, r in enumerate(self.candidate_table()) if r["feasible"]]

    # ------------------------------------------------------------------
    # action preview (no state mutation)
    # ------------------------------------------------------------------
    def preview_action(self, node: int) -> int:
        """Mirror of EpisodeEnv16.step's expert forward, on a belief copy.

        Needed because the motion energy must be validated before the parent
        is allowed to move the UAV.  The parent still performs the real step
        (including the real belief update); `self.preview_mismatch_count`
        counts any disagreement between the preview and the executed action
        (deterministic argmax => expected to be 0).
        """
        saved_belief = copy.deepcopy(self.belief)
        try:
            if self.empty_cands:
                belief_feat = EMPTY_BELIEF.copy()
                match_feat = EMPTY_MATCH.copy()
                est = np.asarray([self.pos[0], self.pos[1]], np.float64)
                score, margin = 0.0, 0.0
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
                    if agree:
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
            mtype = self.metrics()[int(node)].node["model_type"]
            if mtype in NORMALIZED:
                belief_feat = ((belief_feat - self.norm["belief"][0]) /
                               self.norm["belief"][1]).astype(np.float32)
                match_feat = ((match_feat - self.norm["match"][0]) /
                              self.norm["match"][1]).astype(np.float32)
                pose_h = ((pose_h - self.norm["history_pose"][0]) /
                          self.norm["history_pose"][1]).astype(np.float32)
            cur_img = self.raster.heading_crop(float(self.pos[0]),
                                               float(self.pos[1]), float(self.yaw))
            with torch.no_grad():
                cur = self.enc4(np.asarray([cur_img], np.uint8))
                batch = {
                    "current": cur,
                    "target_emb": torch.from_numpy(self.gdesc[None].copy()).to(self.device),
                    "history_vis": torch.from_numpy(vis).to(self.device),
                    "history_pose": torch.from_numpy(pose_h).to(self.device),
                    "history_action": torch.from_numpy(act_h).to(self.device),
                    "history_mask": torch.from_numpy(hmask).to(self.device),
                    "belief": torch.from_numpy(belief_feat[None].copy()).to(self.device),
                    "match": torch.from_numpy(match_feat[None].copy()).to(self.device),
                    "landmarks": torch.from_numpy(
                        landmark_inputs(self.belief, self.empty_cands)).to(self.device),
                    "states": torch.from_numpy(
                        np.asarray([[self.pos[0], self.pos[1], self.yaw]],
                                   np.float32)).to(self.device),
                }
                logits = self.models[mtype](batch)
            return int(logits.argmax(-1).item())
        finally:
            self.belief = saved_belief

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def _terminate(self, reason: str, charge_latency=None, charge_energy=0.0):
        self.done = True
        self.success = False
        self.failure = reason
        self.terminated_reason = reason
        if charge_latency is not None:
            self.cumulative_coll_latency_ms += charge_latency["d_coll_ms"]
        if charge_energy:
            self.cumulative_energy_j += charge_energy
            self.remaining_energy_j -= charge_energy

    def step(self, node):
        if self.done:
            raise RuntimeError("step() called on a terminated episode; call start(task)")
        c = self.cfg
        node = int(node)
        table = self.candidate_table()
        n_feasible = sum(r["feasible"] for r in table)
        rec = table[node]

        if n_feasible == 0:
            self._terminate("no_feasible_mec")
            return self._empty_reward(), True, self._info(node, table, None, None,
                                                          n_feasible)
        if not rec["feasible"]:
            if not rec["coverage_valid"]:
                self.constraint_failure = "coverage"
            elif not rec["deadline_valid"]:
                self.constraint_failure = "deadline"
                self.deadline_violation_count += 1
            else:
                self.constraint_failure = "energy"
                self.energy_budget_violation_count += 1
            self.invalid_mec_action_count += 1
            self._terminate("invalid_mec_action")
            return self._empty_reward(), True, self._info(node, table, None, None,
                                                          n_feasible)

        nav_action = self.preview_action(node)
        motion = e_motion_j(nav_action, c)
        if c.enforce_energy and (self.remaining_energy_j
                                 - rec["service_energy_j"]
                                 - motion["e_motion_j"] < 0):
            # service already issued remotely; motion is refused
            self.energy_budget_violation_count += 1
            self._terminate("energy_budget_exhausted",
                            charge_latency=rec["latency"],
                            charge_energy=rec["service_energy_j"])
            return self._empty_reward(), True, self._info(node, table, nav_action,
                                                          rec, n_feasible)

        reward, done, info = super().step(node)
        executed = int(self.steps[-1]["action"])
        if executed != nav_action:
            self.preview_mismatch_count += 1

        coll = e_coll_j(rec["latency"], c)
        motion_exec = e_motion_j(executed, c)
        step_energy = coll["e_coll_j"] + motion_exec["e_motion_j"]
        self.cumulative_coll_latency_ms += rec["latency"]["d_coll_ms"]
        self.cumulative_energy_j += step_energy
        self.remaining_energy_j -= step_energy
        self.model_selection[node] += 1

        self.step_records.append({
            "step": self.step_idx - 1, "selected_mec": node,
            "selected_model": rec["model_type"],
            "navigation_action": executed,
            **{k: rec["latency"][k] for k in
               ("d_enc_ms", "d_up_ms", "d_inf_ms", "d_down_ms", "d_coll_ms")},
            "e_hover_j": coll["e_hover_j"], "e_enc_j": coll["e_enc_j"],
            "e_tx_j": coll["e_tx_j"], "e_rx_j": coll["e_rx_j"],
            "e_forward_j": motion_exec["e_forward_j"],
            "e_turn_j": motion_exec["e_turn_j"],
            "e_total_step_j": step_energy,
            "battery_remaining_j": self.remaining_energy_j,
            "deadline_valid": bool(rec["deadline_valid"]),
            "energy_valid": bool(rec["energy_valid"]),
            "feasible_mec_count": int(n_feasible),
        })

        if done and self.failure:
            self.terminated_reason = self.failure
        return reward, done, self._info(node, table, executed, rec, n_feasible)

    # ------------------------------------------------------------------
    def _empty_reward(self) -> float:
        """Terminal step that produced no service carries no E2E reward.

        The step is a constraint failure, not a latency observation, so the
        navigation reward must not be credited or debited here; the failure is
        signalled through `terminated_reason` / `constraint_failure`.
        """
        return 0.0

    def _info(self, node, table, nav_action, rec, n_feasible) -> dict:
        info = {
            "selected_mec": node, "nav_action": nav_action,
            "feasible_mec_count": int(n_feasible),
            "remaining_energy_j": float(self.remaining_energy_j),
            "energy_budget_j": float(self.energy_budget_j),
            "cumulative_energy_j": float(self.cumulative_energy_j),
            "cumulative_collab_latency_ms": float(self.cumulative_coll_latency_ms),
            "legacy_cumulative_e2e_ms": float(np.sum(self.episode_e2e)),
            "terminated_reason": self.terminated_reason,
            "constraint_failure": self.constraint_failure,
        }
        if rec is not None:
            info.update(rec["latency"])
            info["selected_model"] = rec["model_type"]
        return info
