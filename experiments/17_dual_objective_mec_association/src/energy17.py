#!/usr/bin/env python3
"""Stage 17-A UAV energy model.

Per decision step:

    E_total_step = E_coll + E_motion
    E_coll   = P_hover * D_coll + P_enc * D_enc + P_tx * D_up + P_rx * D_down
    E_motion = E_forward (5 m at v)  |  E_turn (30 deg at omega)  |  0 (Stop)

Flight power uses the full rotary-wing model of Zeng, Xu & Zhang,
"Energy Minimization for Wireless Communication with Rotary-Wing UAV",
IEEE TWC 2019:

    P_fly(v) = P0 (1 + 3 v^2 / Utip^2)
             + Pi * sqrt(1 + v^4 / (4 v0^4))
             + 0.5 * d0 * rho * s * A * v^3

E_forward is NOT approximated as P_hover * t_forward.

All energies are joules.  No auxiliary/board power is added: `p_aux_w`
defaults to 0 so the same device power is never counted twice.
"""
from __future__ import annotations

import math

from config17 import Stage17Config


def hover_power_w(cfg: Stage17Config) -> float:
    """Induced + profile power in hover (W)."""
    return cfg.p0_w + cfg.pi_w


def fly_power_w(v_mps: float, cfg: Stage17Config) -> float:
    """Rotary-wing UAV power at constant forward speed v (W). Zeng 2019."""
    v = max(float(v_mps), 0.0)
    profile = cfg.p0_w * (1.0 + 3.0 * v * v / (cfg.utip_mps ** 2))
    induced = cfg.pi_w * math.sqrt(1.0 + v ** 4 / (4.0 * cfg.v0_mps ** 4))
    parasite = 0.5 * cfg.d0 * cfg.rho * cfg.rotor_s * cfg.rotor_a_m2 * v ** 3
    return profile + induced + parasite


def e_forward_j(cfg: Stage17Config) -> float:
    """Energy of one Forward action: fly 5 m at v_forward."""
    t_forward = cfg.forward_m / cfg.v_forward_mps
    return fly_power_w(cfg.v_forward_mps, cfg) * t_forward


def e_turn_j(cfg: Stage17Config) -> float:
    """Energy of one Turn action: hover while yawing 30 deg at omega."""
    t_turn = cfg.turn_deg / cfg.omega_turn_deg_s
    return hover_power_w(cfg) * t_turn


def e_coll_j(latency: dict, cfg: Stage17Config) -> dict:
    """Energy consumed by one collaborative inference service (J)."""
    d_coll_s = latency["d_coll_ms"] / 1e3
    d_enc_s = latency["d_enc_ms"] / 1e3
    d_up_s = latency["d_up_ms"] / 1e3
    d_down_s = latency["d_down_ms"] / 1e3
    e_hover = hover_power_w(cfg) * d_coll_s
    e_enc = cfg.p_enc_w * d_enc_s
    e_tx = cfg.p_tx_w * d_up_s
    e_rx = cfg.p_rx_w * d_down_s
    return {"e_hover_j": e_hover, "e_enc_j": e_enc, "e_tx_j": e_tx,
            "e_rx_j": e_rx, "e_aux_j": cfg.p_aux_w * d_coll_s,
            "e_coll_j": e_hover + e_enc + e_tx + e_rx + cfg.p_aux_w * d_coll_s}


def e_motion_j(nav_action: int, cfg: Stage17Config) -> dict:
    """Motion energy for the executed navigation action (0/1/2/3)."""
    if nav_action == 1:
        return {"e_forward_j": e_forward_j(cfg), "e_turn_j": 0.0,
                "e_motion_j": e_forward_j(cfg)}
    if nav_action in (2, 3):
        return {"e_forward_j": 0.0, "e_turn_j": e_turn_j(cfg),
                "e_motion_j": e_turn_j(cfg)}
    return {"e_forward_j": 0.0, "e_turn_j": 0.0, "e_motion_j": 0.0}  # Stop


def step_energy_j(nav_action: int, latency: dict, cfg: Stage17Config) -> dict:
    """Full per-step breakdown; E_total_step = E_coll + E_motion."""
    coll = e_coll_j(latency, cfg)
    motion = e_motion_j(nav_action, cfg)
    return {**coll, **motion,
            "e_total_step_j": coll["e_coll_j"] + motion["e_motion_j"]}
