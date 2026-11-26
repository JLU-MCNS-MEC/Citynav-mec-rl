#!/usr/bin/env python3
"""Stage 17-A-R2 collaborative inference service latency.

    D_coll(t, m) = D_UAV(t) + D_up(t, m) + D_inf(m)

    D_UAV = D_layer4 + D_layer2 + D_matching      (see compute17)
    D_up  = L_up / R_up(d_h)                     (Stage-17 geometry, 50/10 m)
    D_inf = G_m / 10 TFLOPS                      (measured Stage-14 FLOPs)

The MEC -> UAV action-return delay is neglected by review decision
(2026-11-26), so `include_downlink=False` and D_down = 0 by construction.

The legacy Stage-16 E2E (enc + up with the old 1,605,632-bit payload and the
old 80/10 m geometry) is kept as a comparison metric, never as the Stage-17
objective.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in [HERE, REPO / "experiments/16_cumulative_e2e_association/p1",
          REPO / "experiments/12_hetero_mec_association/scripts"]:
    sys.path.insert(0, str(p))

import protocol as P12                      # noqa: E402
from env16 import MODEL_GFLOPS_MEASURED     # noqa: E402

from comm17 import uplink_latency_ms        # noqa: E402
from compute17 import uav_compute_ms        # noqa: E402
from config17 import Stage17Config          # noqa: E402
from multimodal17 import calculate_uplink_payload_bits  # noqa: E402


def d_inf_ms(model_type: str, cfg: Stage17Config) -> float:
    """MEC inference: measured FLOPs / 10 TFLOPS."""
    return MODEL_GFLOPS_MEASURED[model_type] / cfg.mec_tflops


def d_down_ms(distance_m: float, cfg: Stage17Config) -> float:
    """MEC -> UAV action return. Neglected unless explicitly enabled."""
    if not cfg.include_downlink:
        return 0.0
    import math
    _, snr_up = _downlink_snr(distance_m, cfg)
    rate = cfg.bandwidth_hz * math.log2(1.0 + snr_up)
    return 8.0 * cfg.downlink_payload_bytes / max(rate, 1e-15) * 1e3


def _downlink_snr(distance_m: float, cfg: Stage17Config):
    from comm17 import uplink_rate_bps
    rate, snr = uplink_rate_bps(distance_m, cfg)
    return rate, snr * (cfg.mec_tx_power_w / max(cfg.p_tx_w, 1e-12))


def service_latency_ms(distance_m: float, model_type: str,
                       cfg: Stage17Config, payload: dict | None = None) -> dict:
    """All components of one collaborative service, in milliseconds."""
    if payload is None:
        payload = calculate_uplink_payload_bits(cfg)
    comp = uav_compute_ms(cfg)
    up = uplink_latency_ms(payload["uplink_payload_bits"], distance_m, cfg)
    inf = d_inf_ms(model_type, cfg)
    down = d_down_ms(distance_m, cfg)
    return {
        "d_visual_layer4_ms": comp["d_visual_layer4_ms"],
        "d_visual_layer2_ms": comp["d_visual_layer2_ms"],
        "d_matching_ms": comp["d_matching_ms"],
        "d_matching_verified": comp["d_matching_verified"],
        "d_uav_total_ms": comp["d_uav_total_ms"],
        "d_up_ms": up["uplink_latency_ms"],
        "d_inf_ms": inf,
        "d_down_ms": down,
        "d_coll_ms": comp["d_uav_total_ms"] + up["uplink_latency_ms"] + inf + down,
        "uplink_payload_bits": up["uplink_payload_bits"],
        "uplink_payload_bytes": up["uplink_payload_bytes"],
        "uplink_rate_bps": up["uplink_rate_bps"],
        "uplink_snr": up["uplink_snr"],
        "d_3d_m": up["d_3d_m"],
        "elevation_deg": up["elevation_deg"],
    }


def legacy_stage16_e2e_ms(distance_m: float, model_type: str) -> float:
    """Stage-16 E2E for the same horizontal distance (comparison only).

    Uses the Stage-12 payload (1,605,632 bit) and the Stage-12 80/10 m
    geometry, exactly as EpisodeEnv16 did.
    """
    return P12.encode_ms() + P12.comm_latency_ms(distance_m) \
        + MODEL_GFLOPS_MEASURED[model_type] / 10.0
