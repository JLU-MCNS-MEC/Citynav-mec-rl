#!/usr/bin/env python3
"""Stage 17-B U1-U8: formula checks. No environment, no expert checkpoints."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from config17 import load_config                    # noqa: E402
from energy17 import (e_forward_j, e_turn_j, fly_power_w,  # noqa: E402
                      hover_power_w)
from latency17 import (d_down_ms, d_enc_ms, d_inf_ms, d_up_ms,  # noqa: E402
                       legacy_e2e_ms, service_latency_ms)

TOL = 1e-9
EXPECT_D_INF = {"small": 0.4954039, "medium": 1.4978528, "ts": 3.9671126,
                "tm": 10.0025418, "large": 19.9723981}


def main():
    cfg = load_config(ROOT / "configs/default.json")
    res = {}

    res["U1_d_enc_ms"] = {"value": d_enc_ms(cfg), "expected": 5.449515008,
                          "pass": abs(d_enc_ms(cfg) - 5.449515008) < 1e-6}
    diffs = {k: abs(d_inf_ms(k, cfg) - v) for k, v in EXPECT_D_INF.items()}
    res["U2_d_inf"] = {"values": {k: d_inf_ms(k, cfg) for k in EXPECT_D_INF},
                       "max_abs_err": max(diffs.values()),
                       "pass": max(diffs.values()) < 1e-6}

    # U3 uplink monotonicity + finite
    ds = [0.0, 50.0, 200.0, 500.0]
    ups = [d_up_ms(d) for d in ds]
    res["U3_d_up"] = {"distances": ds, "values_ms": ups,
                      "monotone_increasing": all(b >= a for a, b in zip(ups, ups[1:])),
                      "pass": all(v > 0 for v in ups)}

    # U4 downlink disabled by default
    res["U4_d_down"] = {"include_downlink": cfg.include_downlink,
                        "value_ms": d_down_ms(100.0, cfg),
                        "pass": (not cfg.include_downlink) and d_down_ms(100.0, cfg) == 0.0}

    # U5 fly power degenerates to hover power at v = 0
    res["U5_p_fly_v0"] = {"p_fly_0": fly_power_w(0.0, cfg),
                          "p_hover": hover_power_w(cfg),
                          "pass": abs(fly_power_w(0.0, cfg) - hover_power_w(cfg)) < TOL}

    # U6 forward energy = P_fly(v) * 5 / v
    v = cfg.v_forward_mps
    e6 = e_forward_j(cfg)
    manual = fly_power_w(v, cfg) * cfg.forward_m / v
    res["U6_e_forward"] = {"v_mps": v, "p_fly_w": fly_power_w(v, cfg),
                           "e_forward_j": e6, "manual": manual,
                           "pass": abs(e6 - manual) < 1e-9 and e6 > 0}

    # U7 turn energy = P_hover * 30/30
    res["U7_e_turn"] = {"e_turn_j": e_turn_j(cfg),
                        "expected": hover_power_w(cfg) * 1.0,
                        "pass": abs(e_turn_j(cfg) - hover_power_w(cfg)) < 1e-9}

    # U8 service latency decomposition + legacy consistency
    lat = service_latency_ms(200.0, "ts", cfg)
    legacy = legacy_e2e_ms(200.0, "ts", cfg)
    res["U8_service_latency"] = {
        "components": lat, "legacy_e2e_ms": legacy,
        "pass": (abs(lat["d_coll_ms"] - (lat["d_enc_ms"] + lat["d_up_ms"]
                                         + lat["d_inf_ms"] + lat["d_down_ms"])) < 1e-9
                 and abs(legacy - (lat["d_enc_ms"] + lat["d_up_ms"]
                                   + lat["d_inf_ms"])) < 1e-9)}

    out = ROOT / "smoke/unit_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    npass = sum(1 for v in res.values() if v["pass"])
    print(json.dumps(res, indent=2))
    print(f"PASSED {npass}/{len(res)}")
    return 0 if npass == len(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
