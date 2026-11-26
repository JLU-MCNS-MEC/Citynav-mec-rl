#!/usr/bin/env python3
"""Stage 17-A-R2 formula checks (U1-U10).

Pure arithmetic: no expert checkpoint is loaded, no environment is created and
no episode is executed.  The IEEE flight-power reference is implemented HERE,
independently of `src/energy17.py`, so the two must agree numerically instead
of a function being compared with itself.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from comm17 import geometry, uplink_latency_ms          # noqa: E402
from compute17 import layer2_flops, uav_compute_ms      # noqa: E402
from config17 import load_config                        # noqa: E402
from energy17 import e_forward_j, e_turn_j, fly_power_w, hover_power_w  # noqa: E402
from latency17 import (d_down_ms, d_inf_ms, legacy_stage16_e2e_ms,  # noqa: E402
                       service_latency_ms)
from multimodal17 import calculate_uplink_payload_bits  # noqa: E402

TOL = 1e-6


def ieee_fly_power(v: float, P0=79.86, Pi=88.63, Utip=120.0, v0=4.03,
                   d0=0.6, rho=1.225, s=0.05, A=0.503) -> float:
    """Zeng 2019 eq.(12), written from the paper, independent of energy17."""
    x = v * v / (2.0 * v0 * v0)
    return (P0 * (1.0 + 3.0 * v * v / (Utip * Utip))
            + Pi * (math.sqrt(1.0 + v ** 4 / (4.0 * v0 ** 4))
                    - v * v / (2.0 * v0 * v0)) ** 0.5
            + 0.5 * d0 * rho * s * A * v ** 3)


def main():
    cfg = load_config(ROOT / "configs/default.json")
    res = {}

    comp = uav_compute_ms(cfg)
    res["U1_layer4"] = {"flops": comp["layer4_flops"],
                        "d_ms": comp["d_visual_layer4_ms"],
                        "expected_ms": 5.449515008,
                        "pass": abs(comp["d_visual_layer4_ms"] - 5.449515008) < 1e-6}

    l2, table = layer2_flops(cfg)
    res["U2_layer2"] = {"flops": l2, "expected_flops": 3318841344.0,
                        "d_ms": comp["d_visual_layer2_ms"],
                        "expected_d_ms": 2.212560896,
                        "ops": len(table),
                        "pass": (l2 == 3318841344.0
                                 and abs(comp["d_visual_layer2_ms"] - 2.212560896) < 1e-6)}

    res["U3_uav_total"] = {"d_uav_ms": comp["d_uav_total_ms"],
                           "expected_ms": 7.662075904,
                           "matching_verified": comp["d_matching_verified"],
                           "pass": abs(comp["d_uav_total_ms"] - 7.662075904) < 1e-6}

    exp_d_inf = {"small": 0.4954039, "medium": 1.4978528, "ts": 3.9671126,
                 "tm": 10.0025418, "large": 19.9723981}
    err = max(abs(d_inf_ms(k, cfg) - v) for k, v in exp_d_inf.items())
    res["U4_d_inf"] = {"values": {k: d_inf_ms(k, cfg) for k in exp_d_inf},
                       "max_abs_err": err, "pass": err < 1e-6}

    # IEEE reference points, computed independently
    p3, p5 = ieee_fly_power(3.0), ieee_fly_power(5.0)
    res["U5_fly_power_independent"] = {
        "p3_independent": p3, "p3_implementation": fly_power_w(3.0, cfg),
        "p5_independent": p5, "p5_implementation": fly_power_w(5.0, cfg),
        "expected_p3": 157.56, "expected_p5": 143.61,
        "pass": (abs(p3 - 157.55548) < 1e-3 and abs(p5 - 143.61354) < 1e-3
                 and abs(fly_power_w(3.0, cfg) - p3) < 1e-9
                 and abs(fly_power_w(5.0, cfg) - p5) < 1e-9)}

    e3 = fly_power_w(3.0, cfg) * 5.0 / 3.0
    e5 = fly_power_w(5.0, cfg) * 5.0 / 5.0
    res["U6_forward_energy"] = {
        "e_forward_v3_j": e3, "expected": 262.5925,
        "e_forward_v5_j": e5, "expected_v5": 143.61354,
        "e_forward_default_cfg_j": e_forward_j(cfg),
        "pass": abs(e3 - 262.5925) < 1e-3 and abs(e5 - 143.61354) < 1e-3}

    res["U7_turn_hover"] = {"hover_w": hover_power_w(cfg),
                            "expected_hover_w": 168.49,
                            "e_turn_j": e_turn_j(cfg),
                            "pass": abs(hover_power_w(cfg) - 168.49) < 1e-9
                                    and abs(e_turn_j(cfg) - 168.49) < 1e-9}

    payload = calculate_uplink_payload_bits(cfg)
    res["U8_uplink_payload"] = {
        "total_bits": payload["uplink_payload_bits"],
        "expected_bits": 4281664,
        "total_bytes": payload["uplink_payload_bytes"],
        "per_modality": {m["name"]: m["payload_bits"]
                         for m in payload["modalities"] if m["uploaded"]},
        "not_uploaded": payload["derived_not_uploaded"],
        "pass": payload["uplink_payload_bits"] == 4281664}

    g = geometry(500.0, cfg)
    lat500 = uplink_latency_ms(payload["uplink_payload_bits"], 500.0, cfg)
    res["U9_comm_geometry"] = {
        "d_vertical_m": g["d_vertical_m"], "expected_vertical_m": 40.0,
        "d_3d_m_at_500": g["d_3d_m"],
        "expected_d3d": math.sqrt(500 ** 2 + 40 ** 2),
        "elevation_deg": g["elevation_deg"],
        "uplink_latency_ms_at_500": lat500["uplink_latency_ms"],
        "pass": abs(g["d_vertical_m"] - 40.0) < 1e-9
                and abs(g["d_3d_m"] - math.sqrt(500 ** 2 + 40 ** 2)) < 1e-6}

    s = service_latency_ms(200.0, "ts", cfg, payload)
    res["U10_service_latency"] = {
        "components": {k: s[k] for k in
                       ("d_visual_layer4_ms", "d_visual_layer2_ms", "d_matching_ms",
                        "d_uav_total_ms", "d_up_ms", "d_inf_ms", "d_down_ms",
                        "d_coll_ms")},
        "legacy_stage16_e2e_ms": legacy_stage16_e2e_ms(200.0, "ts"),
        "downlink_disabled": d_down_ms(200.0, cfg) == 0.0,
        "pass": (abs(s["d_coll_ms"]
                     - (s["d_uav_total_ms"] + s["d_up_ms"] + s["d_inf_ms"]
                        + s["d_down_ms"])) < 1e-9
                 and s["d_down_ms"] == 0.0)}

    out = ROOT / "smoke/unit_checks_r2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    n = sum(1 for v in res.values() if v["pass"])
    print(f"PASSED {n}/{len(res)}")
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
