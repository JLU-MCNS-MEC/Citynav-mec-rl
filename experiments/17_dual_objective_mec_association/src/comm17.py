#!/usr/bin/env python3
"""Stage 17 uplink channel model (Stage 17-A-R2).

Same structure as Stage 12 `protocol.uplink_rate_bps` (probabilistic LoS,
free-space path loss, Shannon rate), but with the Stage 17 geometry:

    UAV height            50 m   (Stage 12/16 used 80 m)
    MEC antenna height    10 m
    vertical separation   40 m
    d_3D = sqrt(d_h^2 + 40^2)
    elevation = atan2(40, d_h)   [degrees, matching the Stage-12 formula]
    coverage radius       500 m horizontal

Stage 12's file is NOT modified; this module only reads its constants.
"""
from __future__ import annotations

import math

from config17 import Stage17Config

# inherited constants (Stage 12 protocol.py:119-134)
NOISE_PSD_DBM_HZ = -174.0
NOISE_FIGURE_DB = 9.0
LOS_A, LOS_B = 9.61, 0.16
EXC_LOS_DB, EXC_NLOS_DB = 1.0, 20.0
C_LIGHT = 299_792_458.0


def dbm_to_watt(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


def geometry(distance_m: float, cfg: Stage17Config) -> dict:
    dh = max(float(distance_m), 0.0)
    dv = abs(cfg.uav_height_m - cfg.mec_antenna_height_m)
    d3 = max(math.sqrt(dh * dh + dv * dv), 1.0)
    elevation_deg = math.degrees(math.atan2(dv, max(dh, 1e-15)))
    return {"d_horizontal_m": dh, "d_vertical_m": dv, "d_3d_m": d3,
            "elevation_deg": elevation_deg}


def uplink_rate_bps(distance_m: float, cfg: Stage17Config) -> tuple[float, float]:
    """Return (rate_bps, linear_SNR) at horizontal distance `distance_m`."""
    g = geometry(distance_m, cfg)
    p_los = 1.0 / (1.0 + LOS_A * math.exp(-LOS_B * (g["elevation_deg"] - LOS_A)))
    fsl_db = 20.0 * math.log10(4.0 * math.pi * cfg.carrier_hz * g["d_3d_m"] / C_LIGHT)
    path_loss_db = fsl_db + p_los * EXC_LOS_DB + (1.0 - p_los) * EXC_NLOS_DB
    noise_w = dbm_to_watt(NOISE_PSD_DBM_HZ
                          + 10.0 * math.log10(cfg.bandwidth_hz)
                          + NOISE_FIGURE_DB)
    snr = cfg.p_tx_w * 10.0 ** (-path_loss_db / 10.0) / max(noise_w, 1e-15)
    rate = cfg.bandwidth_hz * math.log2(1.0 + snr)
    return max(rate, 1e-15), snr


def uplink_latency_ms(payload_bits: float, distance_m: float,
                      cfg: Stage17Config) -> dict:
    rate, snr = uplink_rate_bps(distance_m, cfg)
    lat = float(payload_bits) / rate * 1e3
    return {"uplink_payload_bits": float(payload_bits),
            "uplink_payload_bytes": float(payload_bits) / 8.0,
            "uplink_rate_bps": float(rate), "uplink_snr": float(snr),
            "uplink_latency_ms": lat,
            "d_horizontal_m": geometry(distance_m, cfg)["d_horizontal_m"],
            "d_3d_m": geometry(distance_m, cfg)["d_3d_m"],
            "elevation_deg": geometry(distance_m, cfg)["elevation_deg"]}
