#!/usr/bin/env python3
"""Stage 13-R2 preset: concrete architectures close to 5/15/40/100/200 GFLOPs.

No model is modified or trained.  The presets are for confirmation.

Design notes
------------
* Transformers use the canonical 104-token input (16 history tokens).
* Small/Medium keep the late-fusion MLP / goal-cross-attention design and all
  modalities.  To reach 5/15 GFLOPs without absurd D or fusion widths, a
  per-token visual MLP (2048 -> F_token -> 2048) is inserted before pooling /
  cross-attention.  It is an MLP, not a Transformer encoder.
* All models are trained from scratch; inheritance is not considered.
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
REV = HERE.parent
RES = REV / "results"


def lin(i, o):
    return i * o + o


def small(D=512, H=2048, Ftok=12288):
    flops = (49 * 4 * 2048 * Ftok
             + 2 * (4633 * D + (8 * D + 4) * H))
    params = (2 * 2048 * Ftok + Ftok + 2048
              + lin(2048, D) * 2 + lin(512, D) + lin(3, D) + lin(4, D)
              + lin(12, D) + lin(3, D) + lin(3, D)
              + lin(8 * D, H) + lin(H, 4))
    return flops, params


def medium(D=512, Fgc=4096, H=2048, Ftok=36864):
    flops = (49 * 4 * 2048 * Ftok
             + 2 * (103035 * D + 104 * D * D + 2 * D * Fgc
                    + (7 * D + 4) * H))
    params = (2 * 2048 * Ftok + Ftok + 2048
              + lin(2048, D) + lin(512, D) + 4 * D * D + 4 * D
              + 2 * D + lin(D, Fgc) + lin(Fgc, D)
              + lin(2048, D) + lin(3, D) + lin(4, D) + lin(12, D)
              + lin(3, D) + lin(3, D) + lin(7 * D, H) + lin(H, 4))
    return flops, params


def transformer(D, L, F, Fgc, T=104, tm=False):
    if tm:
        flops = 2 * (133804 * D + 53 * D * D
                     + L * (4 * T * D * D + 2 * T * T * D + 2 * T * D * F))
        p = (lin(2048, D) + 49 * D + lin(512, D) + lin(2048, D)
             + lin(3, D) + lin(D, D) + 4 * D + 16 * D
             + lin(12, D) + lin(D, D) + lin(3, D) + lin(D, D)
             + lin(3, D) + lin(D, D) + 7 * D + D + 2 * D + lin(D, 4)
             + L * (4 * D * D + 4 * D + lin(D, F) + lin(F, D) + 4 * D))
        return flops, p
    flops = 2 * (133966 * D + 100 * D * D + 2 * D * Fgc
                 + L * (4 * T * D * D + 2 * T * T * D + 2 * T * D * F))
    p = (lin(2048, D) + lin(512, D) + 4 * D * D + 4 * D
         + 2 * D + lin(D, Fgc) + lin(Fgc, D)
         + lin(2048, D) + lin(3, D) + lin(4, D) + lin(12, D)
         + lin(3, D) + lin(3, D) + lin(D, 4) + 20 * D
         + L * (4 * D * D + 4 * D + lin(D, F) + lin(F, D) + 4 * D))
    return flops, p


PRESETS = [
    # model, architecture, D, L, heads, F, extra, flops, params
    ("Small", "MLP + per-token MLP", 512, 0, 0, 2048, 12288),
    ("Medium", "goal cross-attn + per-token MLP", 512, 0, 8, 4096, 36864),
    ("Transformer-S", "104-token Transformer", 1408, 8, 8, 5376, 0),
    ("Transformer-M", "104-token Transformer", 2048, 10, 16, 7424, 0),
    ("Large", "104-token Transformer", 2560, 12, 16, 10240, 0),
]
TARGET = {"Small": 5e9, "Medium": 15e9, "Transformer-S": 40e9,
          "Transformer-M": 100e9, "Large": 200e9}


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, arch, D, L, heads, F, extra in PRESETS:
        if model == "Small":
            flops, params = small(D=D, H=F, Ftok=extra)
        elif model == "Medium":
            flops, params = medium(D=D, Fgc=F, H=2048, Ftok=extra)
        elif model == "Transformer-M":
            flops, params = transformer(D, L, F, F, tm=True)
        else:
            flops, params = transformer(D, L, F, 3 * D if model == "Transformer-S"
                                        else 4 * D)
        target = TARGET[model]
        rows.append({
            "model": model, "architecture": arch, "hidden": D,
            "layers": L if L else "n/a", "heads": heads if heads else "n/a",
            "ffn": F, "extra_mlp": extra if extra else "n/a",
            "predicted_gflops": flops / 1e9, "target_gflops": target / 1e9,
            "deviation_pct": 100 * (flops - target) / target,
            "params_m": params / 1e6,
        })
        print(f"{model:14s} {arch:42s} D={D:5d} L={L:2d} H={heads:2d} "
              f"F={F:6d} -> {flops/1e9:7.2f} GFLOPs "
              f"({100*(flops-target)/target:+6.2f}%) params={params/1e6:7.1f}M")
    with (RES / "preset_scheme.csv").open("w", newline="",
                                         encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {RES/'preset_scheme.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
