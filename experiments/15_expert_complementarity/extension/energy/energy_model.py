#!/usr/bin/env python3
"""Stage 15-Extension F: UAV propulsion + onboard energy model.

Propulsion power follows Zeng, Xu & Zhang (IEEE TWC 2019, DOI
10.1109/TWC.2019.2902559).  Platform coefficients are the paper's reference
values and are NOT measurements of the CityNav UAV.

Action timing (SIMULATION ASSUMPTIONS, to be calibrated):
  Forward : 5.0 m at V_FORWARD = 1.0 m/s   -> 5.0 s
  Turn    : 30 deg at OMEGA_TURN = 30 deg/s -> 1.0 s
  Hover   : during encoding / uplink / MEC inference wait
"""
from __future__ import annotations

import math

# ---- Zeng et al. (2019) reference coefficients ----------------------------
P0 = 79.86          # W, blade profile power in hover
Pi = 88.63          # W, induced power in hover
UTIP = 120.0        # m/s, tip speed of rotor blade
V0 = 4.03           # m/s, mean rotor induced velocity in hover
D0 = 0.6            # fuselage drag ratio
RHO = 1.225         # kg/m^3, air density
S = 0.05            # rotor solidity
A = 0.503           # m^2, rotor disc area

# ---- CityNav action geometry (from closed_loop.py) ------------------------
FORWARD_M = 5.0
TURN_RAD = math.radians(30.0)

# ---- Simulation assumptions (NOT measured; calibrate before paper claims) -
V_FORWARD = 1.0     # m/s  (assumption)
OMEGA_TURN = math.radians(30.0)  # rad/s (assumption: 30 deg/s)
P_ENC = 15.0        # W, onboard visual encoder (assumption)
P_TX = 0.1          # W, uplink RF (Stage 12)
P_AUX = 5.0         # W, avionics / gimbal / compute board (assumption)
BATTERY_WH = 100.0  # Wh, assumed pack for budget tests
BATTERY_J = BATTERY_WH * 3600.0


#: Model mode (Stage 15-Extension, 2026-11-25)
#:   "compute"    E = E_enc + E_comm + E_hover(t_infer)      <- default
#:   "propulsion" full Zeng model including flight (legacy reference)
#: The compute mode is the one the association action can actually change:
#: encoding time is fixed, communication and inference depend on the node.
MODEL_MODE = "compute"
INCLUDE_PROPULSION = False


def configure(v_forward: float | None = None,
              omega_turn_deg: float | None = None,
              include_propulsion: bool | None = None) -> None:
    """Override the simulation kinematics assumptions (sensitivity analysis).

    The CityNav protocol defines discrete 5 m / 30 deg actions and no speeds;
    v and omega are therefore free parameters that must be calibrated against
    a real platform before any energy claim is reported.
    """
    global V_FORWARD, OMEGA_TURN, MODEL_MODE, INCLUDE_PROPULSION
    if v_forward:
        V_FORWARD = float(v_forward)
    if omega_turn_deg:
        OMEGA_TURN = math.radians(float(omega_turn_deg))
    if include_propulsion is not None:
        INCLUDE_PROPULSION = bool(include_propulsion)
        MODEL_MODE = "propulsion" if INCLUDE_PROPULSION else "compute"


def propulsion_power(v: float) -> float:
    """Rotor-wing UAV propulsion power at speed v (m/s)."""
    if v <= 1e-9:
        return P0 + Pi
    return (P0 * (1.0 + 3.0 * v * v / (UTIP * UTIP))
            + Pi * math.sqrt(1.0 + v ** 4 / (4.0 * V0 ** 4))
            + 0.5 * D0 * RHO * S * A * v ** 3)


def hover_power() -> float:
    return P0 + Pi


def step_timing(action: int) -> dict:
    """Return per-step time components (seconds) for one navigation action."""
    if action == 1:
        t_move = FORWARD_M / V_FORWARD
    elif action in (2, 3):
        t_move = TURN_RAD / OMEGA_TURN
    else:
        t_move = 0.0
    return {"t_move": t_move}


def step_energy(action: int, t_enc_s: float, t_comm_s: float,
                t_infer_s: float) -> dict:
    """Energy (J) for one decision step.

    No double counting: the UAV is either moving (propulsion power) or
    hovering (hover power) during the communication/encoding/inference wait.
    Encoder and RF loads are added on top of hover power during that window;
    auxiliary avionics are always on.
    """
    t_enc_s = max(0.0, float(t_enc_s))
    t_comm_s = max(0.0, float(t_comm_s))
    t_infer_s = max(0.0, float(t_infer_s))

    if not INCLUDE_PROPULSION:
        # --- compute / communication / hover-while-waiting only -------------
        # The UAV is airborne but flight energy is explicitly out of scope for
        # this experiment (user decision, 2026-11-25).
        e_enc = P_ENC * t_enc_s
        e_comm = P_TX * t_comm_s
        e_hover = hover_power() * t_infer_s
        return {
            "t_enc_s": t_enc_s, "t_comm_s": t_comm_s, "t_infer_s": t_infer_s,
            "e_enc_j": e_enc, "e_comm_j": e_comm, "e_hover_j": e_hover,
            "e_move_j": 0.0, "e_aux_j": 0.0,
            "e_wait_j": e_enc + e_comm + e_hover,
            "e_total_j": e_enc + e_comm + e_hover,
        }

    # --- legacy Zeng propulsion model (reference only) ----------------------
    t = step_timing(action)
    t_move = t["t_move"]
    t_wait = t_enc_s + t_comm_s + t_infer_s
    if action == 1:
        E_move = propulsion_power(V_FORWARD) * t_move
    else:
        E_move = hover_power() * t_move
    E_wait = (hover_power() + P_ENC + P_TX) * t_wait
    E_aux = P_AUX * (t_move + t_wait)
    return {
        "t_move_s": t_move, "t_wait_s": t_wait,
        "e_move_j": E_move, "e_wait_j": E_wait, "e_aux_j": E_aux,
        "e_total_j": E_move + E_wait + E_aux,
    }


def episode_energy(steps: list[dict]) -> dict:
    """Aggregate step-level energies into an episode total."""
    tot = {"e_move_j": 0.0, "e_wait_j": 0.0, "e_aux_j": 0.0,
           "e_total_j": 0.0, "t_move_s": 0.0, "t_wait_s": 0.0}
    for s in steps:
        for k in tot:
            tot[k] += s.get(k, 0.0)
    return tot


if __name__ == "__main__":
    import json
    print(json.dumps({
        "propulsion_power_1mps_w": propulsion_power(V_FORWARD),
        "hover_power_w": hover_power(),
        "forward_step_energy_j": step_energy(1, 0.005450, 0.03, 0.004),
        "turn_step_energy_j": step_energy(2, 0.005450, 0.03, 0.004),
        "battery_j": BATTERY_J,
    }, indent=2))
