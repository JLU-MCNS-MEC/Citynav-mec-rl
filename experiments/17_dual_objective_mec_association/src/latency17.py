#!/usr/bin/env python3
"""Stage 17-A collaborative inference service latency.

    D_coll(t, m) = D_enc + D_up(t, m) + D_inf(m) + D_down(t, m)

Encoding / uplink / inference are INHERITED from Stage 12/16 and must not be
re-derived here.  Downlink is new (Stage 16 has none) and uses the *same*
channel model with a different transmit power, plus an explicit payload size.

Units: every function returns milliseconds.  Payload sizes are bits.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "experiments/12_hetero_mec_association/scripts"))
sys.path.insert(0, str(REPO / "experiments/16_cumulative_e2e_association/p1"))

import protocol as P12                      # noqa: E402  (frozen Stage-12 model)
from env16 import MODEL_GFLOPS_MEASURED     # noqa: E402  (measured, 2xMACs)

from config17 import Stage17Config          # noqa: E402


def d_enc_ms(cfg: Stage17Config) -> float:
    """UAV-side visual encoding. INHERITED: encoder FLOPs (Stage 12/16)."""
    return P12.encoder_flops() / cfg.uav_tflops * 1e3


def d_up_ms(distance_m: float) -> float:
    """UAV -> MEC uplink. INHERITED: Stage-12 link model, payload 1,605,632 bit."""
    return P12.comm_latency_ms(distance_m)


def d_inf_ms(model_type: str, cfg: Stage17Config) -> float:
    """MEC inference. INHERITED measured FLOPs / 10 TFLOPS."""
    return MODEL_GFLOPS_MEASURED[model_type] / cfg.mec_tflops


def d_down_ms(distance_m: float, cfg: Stage17Config) -> float:
    """MEC -> UAV action return.

    REVIEW 2026-11-26: neglected by default (action-return delay and the
    hover energy it would imply are ignored, with literature support).  The
    model below stays available for sensitivity studies when
    `include_downlink=True`.

    When enabled it reuses the Stage-12 path-loss/SNR model unchanged; only
    the transmit power differs (MEC instead of UAV).  Because SNR is linear in
    transmit power at fixed geometry and bandwidth,

        SNR_down = SNR_up * (P_mec / P_uav)

    so the Stage-12 rate function is reused rather than re-implemented.
    """
    if not cfg.include_downlink:
        return 0.0
    _, snr_up = P12.uplink_rate_bps(distance_m)
    snr_down = snr_up * (cfg.mec_tx_power_w / P12.TX_POWER_W)
    rate_bps = max(cfg.bandwidth_hz * math.log2(1.0 + snr_down), 1e-15)
    payload_bits = 8.0 * cfg.downlink_payload_bytes
    return payload_bits / rate_bps * 1e3


def service_latency_ms(distance_m: float, model_type: str,
                       cfg: Stage17Config) -> dict:
    """All four components plus the total, in milliseconds."""
    enc = d_enc_ms(cfg)
    up = d_up_ms(distance_m)
    inf = d_inf_ms(model_type, cfg)
    down = d_down_ms(distance_m, cfg)
    return {"d_enc_ms": enc, "d_up_ms": up, "d_inf_ms": inf,
            "d_down_ms": down, "d_coll_ms": enc + up + inf + down}


def legacy_e2e_ms(distance_m: float, model_type: str,
                  cfg: Stage17Config) -> float:
    """Stage-16 definition (enc + up + inf, no downlink), kept for comparison."""
    return (d_enc_ms(cfg) + d_up_ms(distance_m)
            + d_inf_ms(model_type, cfg))
