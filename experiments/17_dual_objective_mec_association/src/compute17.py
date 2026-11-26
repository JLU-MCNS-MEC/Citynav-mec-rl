#!/usr/bin/env python3
"""Stage 17-A-R2 UAV-side compute cost.

Two frozen visual paths run on the UAV every step (both are present in
Stage 16; Stage 17 must charge both):

  layer4  FrozenEncoder: torchvision resnet50 IMAGENET1K_V2 (fp16), 224x224,
          produces the 49x2048 tokens consumed by the navigation experts.
          FLOPs are the measured 8,174,272,512 from
          experiments/12.../results/protocol/encoder_flops.json.

  layer2  matching.Encoder stem = conv1 + bn1 + relu + maxpool + layer1 +
          layer2 of the same resnet50 (fp32), 224x224, used only for UAV-local
          visual matching.  The full-model number (8.174 G) must NOT be reused
          here: the table below computes the stem from its own conv geometry.

  matching  subpixel correlation + belief update.  Its FLOPs are NOT yet
          verified, so the field is an explicit, unset interface
          (`matching_flops = None`).  When unset, D_matching is reported as
          0 ms **with** `d_matching_verified = False`; it is never silently
          assumed to be zero.

FLOP convention: 2 x MACs, convolution only (no BatchNorm/ReLU/Softmax), the
same convention as the Stage-16 preflight audit.
"""
from __future__ import annotations

from config17 import Stage17Config

# (name, Cin, Cout, kernel, out_h, out_w, groups=1)
RESNET50_STEM_OPS = [
    ("conv1",            3,   64, 7, 112, 112),
    ("layer1.0.conv1",   64,  64, 1,  56,  56),
    ("layer1.0.conv2",   64,  64, 3,  56,  56),
    ("layer1.0.conv3",   64, 256, 1,  56,  56),
    ("layer1.0.downsample", 64, 256, 1, 56, 56),
    ("layer1.1.conv1",   64,  64, 1,  56,  56),
    ("layer1.1.conv2",   64,  64, 3,  56,  56),
    ("layer1.1.conv3",   64, 256, 1,  56,  56),
    ("layer1.2.conv1",   64,  64, 1,  56,  56),
    ("layer1.2.conv2",   64,  64, 3,  56,  56),
    ("layer1.2.conv3",   64, 256, 1,  56,  56),
    ("layer2.0.conv1",  256, 128, 1,  28,  28),
    ("layer2.0.conv2",  128, 128, 3,  28,  28),
    ("layer2.0.conv3",  128, 512, 1,  28,  28),
    ("layer2.0.downsample", 256, 512, 1, 28, 28),
    ("layer2.1.conv1",  512, 128, 1,  28,  28),
    ("layer2.1.conv2",  128, 128, 3,  28,  28),
    ("layer2.1.conv3",  128, 512, 1,  28,  28),
    ("layer2.2.conv1",  512, 128, 1,  28,  28),
    ("layer2.2.conv2",  128, 128, 3,  28,  28),
    ("layer2.2.conv3",  128, 512, 1,  28,  28),
    ("layer2.3.conv1",  512, 128, 1,  28,  28),
    ("layer2.3.conv2",  128, 128, 3,  28,  28),
    ("layer2.3.conv3",  128, 512, 1,  28,  28),
]


def layer2_flops(cfg: Stage17Config) -> tuple[float, list[dict]]:
    """Theoretical conv FLOPs (2 x MACs) of the matching-encoder stem."""
    rows, total = [], 0.0
    for name, cin, cout, k, h, w in RESNET50_STEM_OPS:
        ops = 2.0 * cin * cout * k * k * h * w
        rows.append({"op": name, "macs": ops / 2.0, "flops_2xmacs": ops})
        total += ops
    return total, rows


def layer4_flops(cfg: Stage17Config) -> float:
    """Measured full-model forward to layer4 (Stage-12 encoder_flops.json)."""
    return float(cfg.layer4_flops)


def uav_compute_ms(cfg: Stage17Config) -> dict:
    l4 = layer4_flops(cfg)
    l2, _ = layer2_flops(cfg)
    matching = cfg.matching_flops
    verified = matching is not None
    tflops = cfg.uav_tflops
    d_l4 = l4 / tflops * 1e3
    d_l2 = l2 / tflops * 1e3
    d_m = (float(matching) / tflops * 1e3) if verified else 0.0
    return {
        "layer4_flops": l4,
        "layer2_flops": l2,
        "matching_flops": float(matching) if verified else None,
        "d_visual_layer4_ms": d_l4,
        "d_visual_layer2_ms": d_l2,
        "d_matching_ms": d_m,
        "d_matching_verified": verified,
        "d_uav_total_ms": d_l4 + d_l2 + d_m,
    }
