#!/usr/bin/env python3
"""Stage 17-A-R2 dual-objective constrained MEC association environment.

Subclasses Stage 16's `EpisodeEnv16` without modifying it.  The R2 refactor
splits action *generation* from state *transition* so that every decision step
runs exactly one visual encoding and one expert forward:

    perceive()            layer4 tokens + layer2 matching + belief update
    decide(node)          one expert forward -> navigation action
    transition(action)    movement / collision / stop / history / step bookkeeping

The parent's `step()` is no longer called (it would re-run perception and the
expert forward).  Its navigation semantics are reproduced 1:1 in
`transition()`; the Stage 17-B consistency test compares full trajectories
against `EpisodeEnv16` to prove the equivalence.

Constraints (review 2026-11-26): only
  (C1) episode energy <= battery capacity,
  (C2) per-service D_coll <= d_max_ms.
SR is an objective, not a constraint.
"""
from __future__ import annotations

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

from env16 import (AGREE_M, BEL, CAL, EMPTY_BELIEF, EMPTY_MATCH,  # noqa: E402
                   FORWARD_M, STOP_RADIUS, TAU, TURN_RAD, EpisodeEnv16,
                   belief_features, history_inputs, landmark_inputs,
                   matching, pool_history_from_current, segment_status,
                   subpixel_match)
from compute17 import uav_compute_ms                    # noqa: E402
from config17 import Stage17Config                      # noqa: E402
from energy17 import e_coll_j, e_motion_j, step_energy_j  # noqa: E402
from latency17 import legacy_stage16_e2e_ms, service_latency_ms  # noqa: E402
from multimodal17 import calculate_uplink_payload_bits  # noqa: E402

NORMALIZED_MODELS = {"ts", "tm", "large"}


class EpisodeEnv17(EpisodeEnv16):
    def __init__(self, by_map, device, models, norm, cfg: Stage17Config, **kw):
        kw.setdefault("record_steps", True)
        super().__init__(by_map, device, models, norm, reward_variant="A", **kw)
        self.cfg = cfg
        self.payload = calculate_uplink_payload_bits(cfg)
        self.uav_compute = uav_compute_ms(cfg)

    # ------------------------------------------------------------------
    def start(self, task):
        super().start(task)
        c = self.cfg
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
        self.model_selection = np.zeros(len(self.nodes), dtype=np.int64)
        # R3 event counters (see docs/05 §5)
        self.service_attempt_count = 0
        self.completed_inference_count = 0
        self.executed_navigation_action_count = 0

    # ------------------------------------------------------------------
    def candidate_table(self) -> list[dict]:
        c = self.cfg
        rows = []
        for i, m in enumerate(self.metrics()):
            lat = service_latency_ms(m.distance, m.node["model_type"], c, self.payload)
            cost = e_coll_j(lat, c)["e_coll_j"]
            cov = (m.distance <= c.coverage_m) if c.enforce_coverage else True
            dead = (lat["d_coll_ms"] <= c.d_max_ms) if c.enforce_deadline else True
            energy = (cost <= self.remaining_energy_j) if c.enforce_energy else True
            rows.append({"node": i, "gbs_id": m.node["gbs_id"],
                         "model_type": m.node["model_type"],
                         "distance_m": float(m.distance),
                         "coverage_valid": bool(cov),
                         "deadline_valid": bool(dead),
                         "energy_valid": bool(energy),
                         "feasible": bool(cov and dead and energy),
                         "latency": lat, "service_energy_j": cost})
        return rows

    def masks(self) -> dict:
        t = self.candidate_table()
        f = lambda k: np.asarray([r[k] for r in t], np.float32)  # noqa: E731
        return {
            "coverage_valid_mask": f("coverage_valid"),
            "deadline_valid_mask": f("deadline_valid"),
            "energy_valid_mask": f("energy_valid"),
            "feasible_mec_mask": f("feasible"),
            "candidate_service_latency_ms": np.asarray(
                [r["latency"]["d_coll_ms"] for r in t], np.float32),
            "candidate_energy_cost_j": np.asarray(
                [r["service_energy_j"] for r in t], np.float32),
        }

    def feasible_node_indices(self) -> list[int]:
        return [i for i, r in enumerate(self.candidate_table()) if r["feasible"]]

    # ------------------------------------------------------------------
    def perceive(self) -> dict:
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
        with torch.no_grad():
            cur = self.enc4(np.asarray([cur_img], np.uint8))
        return {"belief_feat": belief_feat, "match_feat": match_feat,
                "vis": vis, "pose_h": pose_h, "act_h": act_h, "hmask": hmask,
                "cur": cur}

    def decide(self, node: int, percept: dict) -> dict:
        mtype = self.metrics()[int(node)].node["model_type"]
        belief_feat = percept["belief_feat"]
        match_feat = percept["match_feat"]
        pose_h = percept["pose_h"]
        if mtype in NORMALIZED_MODELS:
            belief_feat = ((belief_feat - self.norm["belief"][0]) /
                           self.norm["belief"][1]).astype(np.float32)
            match_feat = ((match_feat - self.norm["match"][0]) /
                          self.norm["match"][1]).astype(np.float32)
            pose_h = ((pose_h - self.norm["history_pose"][0]) /
                      self.norm["history_pose"][1]).astype(np.float32)
        batch = {
            "current": percept["cur"],
            "target_emb": torch.from_numpy(self.gdesc[None].copy()).to(self.device),
            "history_vis": torch.from_numpy(percept["vis"]).to(self.device),
            "history_pose": torch.from_numpy(pose_h).to(self.device),
            "history_action": torch.from_numpy(percept["act_h"]).to(self.device),
            "history_mask": torch.from_numpy(percept["hmask"]).to(self.device),
            "belief": torch.from_numpy(belief_feat[None].copy()).to(self.device),
            "match": torch.from_numpy(match_feat[None].copy()).to(self.device),
            "landmarks": torch.from_numpy(
                landmark_inputs(self.belief, self.empty_cands)).to(self.device),
            "states": torch.from_numpy(
                np.asarray([[self.pos[0], self.pos[1], self.yaw]],
                           np.float32)).to(self.device),
        }
        with torch.no_grad():
            logits = self.models[mtype](batch)
        return {"nav_action": int(logits.argmax(-1).item()), "logits": logits}

    # ------------------------------------------------------------------
    def transition(self, action: int, percept: dict) -> dict:
        """Mirror of EpisodeEnv16.step's navigation semantics (no perception)."""
        a = int(action)
        self.hist[0].append(pool_history_from_current(percept["cur"])[0].astype(np.float32))
        self.hist[1].append((float(self.pos[0]), float(self.pos[1]), float(self.yaw)))
        self.hist[2].append(a)
        dist = float(np.linalg.norm(self.pos - self.target))
        self.min_dist = min(self.min_dist, dist)
        executed, aborted = False, False
        if a == 0:
            executed = True
            self.done = True
            self.success = bool(dist <= STOP_RADIUS)
            self.failure = "success" if self.success else "premature_stop"
            self.stop_dist = dist
        elif a == 1:
            nxt = self.pos + FORWARD_M * np.asarray([math.cos(self.yaw),
                                                     math.sin(self.yaw)])
            st = segment_status(self.info, float(self.pos[0]), float(self.pos[1]),
                                float(nxt[0]), float(nxt[1]))
            if st == "collision":
                self.done, self.failure, self.collision = True, "collision", True
                aborted = True
            elif st == "out_of_bounds":
                self.done, self.failure, self.oob = True, "out_of_bounds", True
                aborted = True
            else:
                self.pos = nxt
                self.n_forward += 1
                executed = True
        elif a == 2:
            self.yaw = (self.yaw + TURN_RAD + math.pi) % (2 * math.pi) - math.pi
            executed = True
        elif a == 3:
            self.yaw = (self.yaw - TURN_RAD + math.pi) % (2 * math.pi) - math.pi
            executed = True
        self.step_idx += 1
        if self.step_idx >= self.step_cap:
            self.done = True
            self.failure = self.failure or "step_cap"
        return {"executed": executed, "aborted": aborted}

    # ------------------------------------------------------------------
    def _record_attempt_only(self, node, n_feasible, reason):
        """Service attempt that never produced an inference (R3 counter rules)."""
        self.step_records.append({
            "step": self.step_idx, "selected_mec": int(node) if node is not None else -1,
            "selected_model": "", "navigation_action": -1,
            "d_visual_layer4_ms": 0.0, "d_visual_layer2_ms": 0.0,
            "d_matching_ms": 0.0, "d_matching_verified": True,
            "d_uav_total_ms": 0.0, "d_up_ms": 0.0, "d_inf_ms": 0.0,
            "d_down_ms": 0.0, "d_coll_ms": 0.0,
            "uplink_payload_bits": 0, "uplink_payload_bytes": 0,
            "uplink_rate_bps": 0.0,
            "e_hover_j": 0.0, "e_enc_j": 0.0, "e_tx_j": 0.0, "e_rx_j": 0.0,
            "e_forward_j": 0.0, "e_turn_j": 0.0,
            "e_coll_j": 0.0, "e_motion_j": 0.0, "e_total_step_j": 0.0,
            "battery_remaining_j": float(self.remaining_energy_j),
            "deadline_valid": False, "energy_valid": False, "coverage_valid": False,
            "feasible_mec_count": int(n_feasible),
            "motion_executed": False, "motion_aborted": False,
            "terminated_reason": reason,
            "service_attempt_count": int(self.service_attempt_count),
            "completed_inference_count": int(self.completed_inference_count),
            "executed_navigation_action_count": int(self.executed_navigation_action_count),
        })

    def _terminate(self, reason: str, failure: str | None = None):
        self.done = True
        self.success = False
        self.failure = failure or reason
        self.terminated_reason = reason

    def _record(self, node, rec, nav_action, energy, latency, motion_executed,
                n_feasible, aborted):
        self.step_records.append({
            "step": self.step_idx, "selected_mec": int(node),
            "selected_model": rec["model_type"] if rec else "",
            "navigation_action": int(nav_action) if nav_action is not None else -1,
            "d_visual_layer4_ms": latency["d_visual_layer4_ms"],
            "d_visual_layer2_ms": latency["d_visual_layer2_ms"],
            "d_matching_ms": latency["d_matching_ms"],
            "d_matching_verified": latency["d_matching_verified"],
            "d_uav_total_ms": latency["d_uav_total_ms"],
            "d_up_ms": latency["d_up_ms"], "d_inf_ms": latency["d_inf_ms"],
            "d_down_ms": latency["d_down_ms"], "d_coll_ms": latency["d_coll_ms"],
            "uplink_payload_bits": latency["uplink_payload_bits"],
            "uplink_payload_bytes": latency["uplink_payload_bytes"],
            "uplink_rate_bps": latency["uplink_rate_bps"],
            "e_hover_j": energy["e_hover_j"], "e_enc_j": energy["e_enc_j"],
            "e_tx_j": energy["e_tx_j"], "e_rx_j": energy["e_rx_j"],
            "e_forward_j": energy["e_forward_j"], "e_turn_j": energy["e_turn_j"],
            "e_coll_j": energy["e_coll_j"], "e_motion_j": energy["e_motion_j"],
            "e_total_step_j": energy["e_total_step_j"],
            "battery_remaining_j": float(self.remaining_energy_j),
            "deadline_valid": bool(rec["deadline_valid"]) if rec else False,
            "energy_valid": bool(rec["energy_valid"]) if rec else False,
            "coverage_valid": bool(rec["coverage_valid"]) if rec else False,
            "feasible_mec_count": int(n_feasible),
            "motion_executed": bool(motion_executed),
            "motion_aborted": bool(aborted),
            "terminated_reason": self.terminated_reason,
            "service_attempt_count": int(self.service_attempt_count),
            "completed_inference_count": int(self.completed_inference_count),
            "executed_navigation_action_count": int(self.executed_navigation_action_count),
        })

    def step(self, node):
        if self.done:
            raise RuntimeError("step() on a terminated episode; call start(task)")
        table = self.candidate_table()
        n_feasible = sum(r["feasible"] for r in table)
        # every call is a service attempt, including illegal / infeasible ones
        self.service_attempt_count += 1

        if node is None:
            if n_feasible == 0:
                self._terminate("no_feasible_mec")
            else:
                self.invalid_mec_action_count += 1
                self.constraint_failure = "invalid_action_index"
                self._terminate("invalid_mec_action")
            self._record_attempt_only(None, n_feasible, self.terminated_reason)
            return 0.0, True, {"terminated_reason": self.terminated_reason,
                               "feasible_mec_count": n_feasible}

        node = int(node)
        if node < 0 or node >= len(table):
            self.invalid_mec_action_count += 1
            self.constraint_failure = "invalid_action_index"
            self._terminate("invalid_mec_action")
            self._record_attempt_only(node, n_feasible, self.terminated_reason)
            return 0.0, True, {"terminated_reason": self.terminated_reason,
                               "feasible_mec_count": n_feasible}

        rec = table[node]
        if n_feasible == 0 or not rec["feasible"]:
            if n_feasible == 0:
                self._terminate("no_feasible_mec")
            else:
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
            self._record_attempt_only(node, n_feasible, self.terminated_reason)
            return 0.0, True, {"terminated_reason": self.terminated_reason,
                               "feasible_mec_count": n_feasible,
                               "constraint_failure": self.constraint_failure}

        percept = self.perceive()
        dec = self.decide(node, percept)
        nav_action = dec["nav_action"]
        self.completed_inference_count += 1

        latency = rec["latency"]
        coll = e_coll_j(latency, self.cfg)
        motion = e_motion_j(nav_action, self.cfg)
        affordable = (self.remaining_energy_j
                      - coll["e_coll_j"] - motion["e_motion_j"] >= 0)

        if not affordable:
            # the inference completed -> the MEC was used even though the
            # navigation action will be refused
            self.model_selection[node] += 1
            energy = {**coll, "e_forward_j": 0.0, "e_turn_j": 0.0,
                      "e_motion_j": 0.0, "e_total_step_j": coll["e_coll_j"]}
            self.cumulative_energy_j += energy["e_total_step_j"]
            self.remaining_energy_j -= energy["e_total_step_j"]
            self.cumulative_coll_latency_ms += latency["d_coll_ms"]
            self.energy_budget_violation_count += 1
            # reason first, then the record, so both carry the same termination
            self._terminate("energy_budget_exhausted")
            self._record(node, rec, nav_action, energy, latency, False,
                         n_feasible, False)
            return 0.0, True, {"terminated_reason": self.terminated_reason,
                               "d_coll_ms": latency["d_coll_ms"],
                               "e_total_step_j": energy["e_total_step_j"]}

        trans = self.transition(nav_action, percept)
        # R3: the commanded action is charged in full, even when it ends in a
        # collision / out-of-bounds termination (fixed-action-cost abstraction).
        energy = step_energy_j(nav_action, latency, self.cfg)
        if trans["executed"]:
            self.executed_navigation_action_count += 1
        self.cumulative_energy_j += energy["e_total_step_j"]
        self.remaining_energy_j -= energy["e_total_step_j"]
        self.cumulative_coll_latency_ms += latency["d_coll_ms"]
        self.model_selection[node] += 1
        legacy = legacy_stage16_e2e_ms(rec["distance_m"], rec["model_type"])
        self.episode_e2e.append(legacy)

        if self.done and self.failure:
            self.terminated_reason = self.failure
        self._record(node, rec, nav_action, energy, latency,
                     trans["executed"], n_feasible, trans["aborted"])

        # R3: Stage 17 no longer returns the Stage-16 reward.  The scalar is 0
        # until Stage 17-C defines the dual-objective scalarisation; every
        # component is exported separately so no learner can silently optimise
        # the legacy objective.
        reward = 0.0
        info = {
            "selected_mec": node, "selected_model": rec["model_type"],
            "nav_action": nav_action,
            "d_coll_ms": latency["d_coll_ms"],
            "e_total_step_j": energy["e_total_step_j"],
            "remaining_energy_j": float(self.remaining_energy_j),
            "cumulative_energy_j": float(self.cumulative_energy_j),
            "cumulative_collab_latency_ms": float(self.cumulative_coll_latency_ms),
            "legacy_cumulative_e2e_ms": float(np.sum(self.episode_e2e)),
            "feasible_mec_count": int(n_feasible),
            "terminated_reason": self.terminated_reason,
            # ---- Stage-17 reward interface (docs/05 §6) ----
            "terminal_success_reward": 1.0 if (self.done and self.success) else 0.0,
            "navigation_quality_reward": 0.0,   # not defined until Stage 17-C
            "collaborative_service_latency_cost": (
                latency["d_coll_ms"] / max(self.cfg.d_max_ms, 1e-9)),
            "energy_consumption_cost": (
                energy["e_total_step_j"] / max(self.cfg.task_energy_budget_j, 1e-9)),
            "energy_budget_violation": int(
                self.terminated_reason == "energy_budget_exhausted"),
            "deadline_violation": int(self.constraint_failure == "deadline"),
            "legacy_stage16_reward": -legacy / 100.0,
            "service_attempt_count": self.service_attempt_count,
            "completed_inference_count": self.completed_inference_count,
            "executed_navigation_action_count": self.executed_navigation_action_count,
        }
        return reward, self.done, info
