#!/usr/bin/env python3
"""Stage 17-A configuration.

Every numeric field carries one of three provenance tags in PARAMETERS.md:

  INHERITED  copied from Stage 12/14/16 without modification
  LITERATURE taken from a published model (Zeng et al. 2019)
  PROVISIONAL introduced here and explicitly flagged for review

Nothing in this file is measured by Stage 17-A: the stage performs no
simulation, training or evaluation.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Stage17Config:
    # ---- inherited kinematics (Stage 12 protocol / CityNav) -------------
    forward_m: float = 5.0            # INHERITED protocol.FORWARD_M
    turn_deg: float = 30.0            # INHERITED protocol.TURN_RAD
    stop_radius_m: float = 20.0       # INHERITED protocol.STOP_RADIUS
    max_steps: int = 200              # INHERITED protocol.MAX_STEPS
    # ---- flight kinematics (PROVISIONAL, user selected v = 3 m/s) --------
    v_forward_mps: float = 3.0        # PROVISIONAL (backup 5.0)
    omega_turn_deg_s: float = 30.0    # PROVISIONAL

    # ---- compute ---------------------------------------------------------
    uav_tflops: float = 1.5e12        # INHERITED protocol.UAV_TFLOPS
    mec_tflops: float = 10.0          # INHERITED env16.MEC_TFLOPS
    coverage_m: float = 500.0         # INHERITED env16.COVERAGE_M

    # ---- propulsion model, Zeng et al. IEEE TWC 2019 ---------------------
    p0_w: float = 79.86               # LITERATURE
    pi_w: float = 88.63               # LITERATURE
    utip_mps: float = 120.0           # LITERATURE
    v0_mps: float = 4.03              # LITERATURE
    d0: float = 0.6                   # LITERATURE
    rho: float = 1.225                # LITERATURE (ISA sea level)
    rotor_s: float = 0.05             # LITERATURE
    rotor_a_m2: float = 0.503         # LITERATURE

    # ---- onboard power ---------------------------------------------------
    p_enc_w: float = 15.0             # LITERATURE (citation to be inserted)
    p_tx_w: float = 0.1               # INHERITED protocol.TX_POWER_W
    p_rx_w: float = 0.0               # irrelevant while include_downlink=False
    p_aux_w: float = 0.0              # no separate board-level term

    # ---- energy budget ---------------------------------------------------
    battery_capacity_wh: float = 100.0     # PROVISIONAL reference pack
    initial_energy_j: float = 360000.0     # = 100 Wh
    # REVIEW 2026-11-26: the only energy constraint is "episode energy does
    # not exceed the battery capacity"; there is no separate task budget.
    task_energy_budget_j: float = 360000.0
    battery_safety_margin_j: float = 0.0

    # ---- per-service deadline -------------------------------------------
    d_max_ms: float = 150.0           # PROVISIONAL: see docs/04

    # ---- downlink --------------------------------------------------------
    # REVIEW 2026-11-26: the MEC->UAV action-return delay (and the hover energy
    # during it) is neglected, per the reviewer's literature support.  The
    # implementation stays available behind the flag for sensitivity studies.
    include_downlink: bool = False
    downlink_payload_bytes: int = 8   # only used when include_downlink=True
    mec_tx_power_w: float = 0.1       # only used when include_downlink=True
    bandwidth_hz: float = 5.0e6       # INHERITED protocol.BANDWIDTH_HZ

    # ---- misc ------------------------------------------------------------
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
