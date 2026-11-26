#!/usr/bin/env python3
"""Stage 17 configuration (Stage 17-A-R2).

Provenance tags used in docs/11_PARAMETERS_REFERENCES.md:

  INHERITED   copied from Stage 12 / 14 / 16 unchanged
  LITERATURE  published value (Zeng 2019, DJI, IEEE Asilomar 2023, ...)
  PROVISIONAL introduced here, explicitly flagged for review
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Stage17Config:
    # ---- navigation (INHERITED) -----------------------------------------
    forward_m: float = 5.0
    turn_deg: float = 30.0
    stop_radius_m: float = 20.0
    max_steps: int = 200

    # ---- flight kinematics (user-selected) -------------------------------
    v_forward_mps: float = 3.0
    omega_turn_deg_s: float = 30.0

    # ---- compute (INHERITED) ---------------------------------------------
    uav_tflops: float = 1.5e12
    mec_tflops: float = 10.0
    layer4_flops: float = 8174272512.0     # measured, encoder_flops.json
    matching_flops: float | None = None    # NOT verified -> None (see compute17)

    # ---- deployment (INHERITED) ------------------------------------------
    coverage_m: float = 500.0

    # ---- communication geometry (Stage 17-A-R2) ---------------------------
    uav_height_m: float = 50.0             # LITERATURE (IEEE TWC 2021)
    mec_antenna_height_m: float = 10.0     # LITERATURE
    carrier_hz: float = 2.0e9              # INHERITED
    bandwidth_hz: float = 5.0e6            # INHERITED
    p_tx_w: float = 0.1                    # INHERITED protocol.TX_POWER_W
    feature_bits: int = 32                 # serialisation precision (FP32)

    # ---- propulsion (Zeng et al. IEEE TWC 2019) ---------------------------
    p0_w: float = 79.86
    pi_w: float = 88.63
    utip_mps: float = 120.0
    v0_mps: float = 4.03
    d0: float = 0.6
    rho: float = 1.225
    rotor_s: float = 0.05
    rotor_a_m2: float = 0.503

    # ---- onboard power ----------------------------------------------------
    p_enc_w: float = 15.0                  # LITERATURE (IEEE GRSL 2026)
    p_aux_w: float = 0.0                   # no separate board-level term

    # ---- energy budget (review 2026-11-26) --------------------------------
    battery_capacity_wh: float = 100.0     # LITERATURE reference (DJI TB47D 99.9 Wh)
    initial_energy_j: float = 360000.0
    task_energy_budget_j: float = 360000.0
    battery_safety_margin_j: float = 0.0

    # ---- per-service deadline --------------------------------------------
    d_max_ms: float = 150.0                # PROVISIONAL (re-check in R2)

    # ---- downlink (review 2026-11-26: neglected) --------------------------
    include_downlink: bool = False
    downlink_payload_bytes: int = 8
    mec_tx_power_w: float = 0.1
    p_rx_w: float = 0.0

    # ---- enforcement flags -------------------------------------------------
    enforce_coverage: bool = True
    enforce_deadline: bool = True
    enforce_energy: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def with_overrides(self, **kw) -> "Stage17Config":
        return replace(self, **kw)


def load_config(path: str | Path | None = None, **overrides) -> Stage17Config:
    cfg = Stage17Config()
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = set(cfg.to_dict())
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        cfg = cfg.with_overrides(**data)
    if overrides:
        cfg = cfg.with_overrides(**overrides)
    return cfg


def dump_config(cfg: Stage17Config, path: str | Path) -> None:
    Path(path).write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
