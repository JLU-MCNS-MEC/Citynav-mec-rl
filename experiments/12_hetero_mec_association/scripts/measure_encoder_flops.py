#!/usr/bin/env python3
"""Measure the UAV visual encoder's FLOPs once (the only measured quantity).

The five MEC experts use the user-frozen modelling compute
(5/15/40/100/200 GFLOPs); they are NOT measured here.  Only the frozen
ResNet-50 layer4 encoder is counted, because ``T_encode`` is derived from the
UAV capacity of 1.5 TFLOPS.

Output: results/protocol/encoder_flops.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P  # noqa: E402

sys.path.insert(0, str(P.REPO / "experiments/05_transformer_m_training/scripts"))
from dataset_v2 import FrozenEncoder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()

    P.PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    enc = FrozenEncoder(a.device)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)

    # warm-up
    with torch.no_grad():
        for _ in range(5):
            enc(np.asarray([img], np.uint8))

    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        enc(np.asarray([img], np.uint8))
    flops = int(fc.get_total_flops())

    torch.cuda.synchronize() if a.device.startswith("cuda") else None
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(a.reps):
            enc(np.asarray([img], np.uint8))
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
    measured_ms = (time.perf_counter() - t0) / a.reps * 1e3

    rec = {
        "encoder": "frozen torchvision resnet50 IMAGENET1K_V2 (layer4, fp16)",
        "input": "1x3x224x224 uint8 (100 m x 100 m heading crop)",
        "flops": flops,
        "flops_giga": flops / 1e9,
        "uav_tflops": P.UAV_TFLOPS / 1e12,
        "encode_ms_model": flops / P.UAV_TFLOPS * 1e3,
        "measured_ms_rtx5090": measured_ms,
        "device": a.device,
        "reps": a.reps,
    }
    out = P.ENCODER_FLOPS_PATH
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
