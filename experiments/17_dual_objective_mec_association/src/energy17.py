#!/usr/bin/env python3
"""Stage 17-A-R2 UAV energy model.

    E_step = E_coll + E_motion
    E_coll = P_hover * D_coll + P_enc * D_UAV + P_tx * D_up (+ P_rx * D_down)

Rotary-wing power, Zeng, Xu & Zhang, IEEE TWC 2019, eq. (12):

    P(v) = P0 (1 + 3 v^2 / Utip^2)
         + Pi * ( sqrt(1 + v^4/(4 v0^4)) - v^2/(2 v0^2) )^{1/2}
         + 0.5 d0 rho s A v^3

The induced term is a square root of a square root.  Stage 17-A used an
incorrect single square root; this file implements the paper's form.  The
R2 unit checks verify the two reference points 157.56 W @ 3 m/s and
143.61 W @ 5 m/s with an independent implementation.

Hover: P_hover = P0 + Pi = 168.49 W.
Turn : T = 30 deg / 30 deg/s = 1 s  ->  E_turn = P_hover.
Forward: E = P(v) * 5 / v  (NOT P_hover * 5 / v).
"""
from __future__ import annotations

import math

from config17 import Stage17Config


def hover_power_w(cfg: Stage17Config) -> float:
    return cfg.p0_w + cfg.pi_w


def fly_power_w(v_mps: float, cfg: Stage17Config) -> float:
    v = max(float(v_mps), 0.0)
    if v == 0.0:
        return hover_power_w(cfg)
    x = v * v / (2.0 * cfg.v0_mps ** 2)
    profile = cfg.p0_w * (1.0 + 3.0 * v * v / (cfg.utip_mps ** 2))
    induced = cfg.pi_w * math.sqrt(math.sqrt(1.0 + x * x) - x)
    parasite = 0.5 * cfg.d0 * cfg.rho * cfg.rotor_s * cfg.rotor_a_m2 * v ** 3
    return profile + induced + parasite


def e_forward_j(cfg: Stage17Config) -> float:
    return fly_power_w(cfg.v_forward_mps, cfg) * (cfg.forward_m / cfg.v_forward_mps)


def e_turn_j(cfg: Stage17Config) -> float:
    return hover_power_w(cfg) * (cfg.turn_deg / cfg.omega_turn_deg_s)


def e_coll_j(latency: dict, cfg: Stage17Config) -> dict:
    """Service energy. `latency` comes from latency17.service_latency_ms."""
    d_coll_s = latency["d_coll_ms"] / 1e3
    d_uav_s = latency["d_uav_total_ms"] / 1e3
    d_up_s = latency["d_up_ms"] / 1e3
    d_down_s = latency.get("d_down_ms", 0.0) / 1e3
    e_hover = hover_power_w(cfg) * d_coll_s
    e_enc = cfg.p_enc_w * d_uav_s
    e_tx = cfg.p_tx_w * d_up_s
    e_rx = cfg.p_rx_w * d_down_s
    e_aux = cfg.p_aux_w * d_coll_s
    return {"e_hover_j": e_hover, "e_enc_j": e_enc, "e_tx_j": e_tx,
            "e_rx_j": e_rx, "e_aux_j": e_aux,
            "e_coll_j": e_hover + e_enc + e_tx + e_rx + e_aux}


def e_motion_j(nav_action: int, cfg: Stage17Config) -> dict:
    """Motion energy of the commanded action.

    R3 decision: whenever the expert emits Forward / TurnLeft / TurnRight the
    full discrete action cost is charged, including when the move triggers a
    collision or an out-of-bounds termination.  This is a fixed-action-cost
    simulation abstraction; the distance flown before the abort is NOT
    modelled (documented in docs/03 and docs/13).

    The only case where motion is not charged is when the remaining budget
    cannot cover it: then the action is refused before execution and only the
    already-issued service is charged (handled by env17.step).
    """
    if nav_action == 0:
        return {"e_forward_j": 0.0, "e_turn_j": 0.0, "e_motion_j": 0.0}
    if nav_action == 1:
        e = e_forward_j(cfg)
        return {"e_forward_j": e, "e_turn_j": 0.0, "e_motion_j": e}
    if nav_action in (2, 3):
        e = e_turn_j(cfg)
        return {"e_forward_j": 0.0, "e_turn_j": e, "e_motion_j": e}
    return {"e_forward_j": 0.0, "e_turn_j": 0.0, "e_motion_j": 0.0}


def step_energy_j(nav_action: int, latency: dict, cfg: Stage17Config) -> dict:
    coll = e_coll_j(latency, cfg)
    motion = e_motion_j(nav_action, cfg)
    return {**coll, **motion,
            "e_total_step_j": coll["e_coll_j"] + motion["e_motion_j"]}
