#!/usr/bin/env python3
"""Verify that every ResNet input at stage 05 is normalised the same way.

Production has ONE normalisation path: uint8 HWC is stacked on the CPU, sent to
the GPU, and converted to float32 / 255 / ImageNet mean-std there.  The cache
pass (history, target reference) and the training loop (current observation)
both use it, so their features are identical by construction.

The script reports:

* ``cache_vs_train_max_abs_diff`` - the cache-pass feature and the training-pass
  feature for the same batch; must be exactly 0
* ``cpu_reference_max_abs_diff`` - the retired CPU-normalisation path against the
  GPU path on the normalised tensor; a few float32 ULP (~1e-6) are expected on
  this platform, which is why the CPU path is no longer used in production
* ``cpu_reference_feature_max_abs_diff`` - how far that ULP difference grows
  after fp16 layer4, i.e. the reason the duplicate implementation was removed
* ``cross_batch_max_abs_diff`` - batch 32 (cache pass) vs batch 128 (training):
  residual fp16/cuDNN kernel difference, not a normalisation difference

Writes results/normalization_check.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset_v2 import (CANDS, CANDIDATE_VERSION, ENCODE_CHUNK, ENCODER_ID,  # noqa: E402
                        FrozenEncoder, RasterStore, load_split)

RESULTS = HERE.parent / "results"
N = 128


def main() -> int:
    episodes = load_split("train_seen")[:6]
    rasters = RasterStore()
    images = []
    for ep in episodes:
        raster = rasters.get(ep.map)
        for x, y, yaw in ep.states[:24]:
            images.append(raster.heading_crop(float(x), float(y), float(yaw)))
        if len(images) >= N:
            break
    images = images[:N]
    print(f"images: {len(images)} crops from {len(episodes)} episodes", flush=True)

    encoder = FrozenEncoder("cuda")
    torch = encoder.torch
    stack = np.stack(images)

    # production path: cache pass (chunked exactly like prepare_v2) vs training
    cache_feats = np.concatenate([encoder.spatial_maps(images[i:i + ENCODE_CHUNK])
                                  for i in range(0, len(images), ENCODE_CHUNK)])
    train_feats = encoder(stack).cpu().numpy()                        # (B,49,2048)
    train_feats = train_feats.reshape(len(images), 7, 7, 2048).transpose(0, 3, 1, 2)
    production_diff = float(np.abs(cache_feats - train_feats).max())
    print(f"cache vs train max_abs_diff = {production_diff:.3e}", flush=True)

    # retired CPU reference path, kept only to document why it was removed
    cpu_norm = encoder.normalise_cpu(images)
    gpu_norm = encoder.normalise_gpu(stack).cpu().numpy()
    cpu_diff = float(np.abs(cpu_norm - gpu_norm).max())
    ulp = float(np.spacing(np.float32(cpu_norm.max())))
    cpu_half = torch.from_numpy(cpu_norm).half().numpy()
    gpu_half = torch.from_numpy(np.ascontiguousarray(gpu_norm)).half().numpy()
    half_mismatch = int((cpu_half != gpu_half).sum())
    cpu_feats = encoder.forward_nchw(torch.from_numpy(cpu_norm).to(encoder.device)).cpu().numpy()
    cpu_feat_diff = float(np.abs(cpu_feats - cache_feats).max())
    print(f"cpu reference normalise max_abs_diff = {cpu_diff:.3e} "
          f"({cpu_diff / max(ulp, 1e-12):.1f} ULP, {half_mismatch} fp16 elements differ)", flush=True)
    print(f"cpu reference feature max_abs_diff = {cpu_feat_diff:.3e}", flush=True)

    # batch-size effect inside the same path (fp16 kernel selection, not normalisation)
    small = encoder.spatial_maps(images[:32])
    cross = float(np.abs(small - cache_feats[:32]).max())
    print(f"cross-batch max_abs_diff (32 vs {ENCODE_CHUNK}) = {cross:.3e} "
          f"(informational; prepare_v2 uses {ENCODE_CHUNK})", flush=True)

    out = {
        "n_images": len(images),
        "sources": [ep.map for ep in episodes],
        "candidate_library": str(CANDS),
        "candidate_version": CANDIDATE_VERSION,
        "encoder": ENCODER_ID,
        "production": "uint8 HWC -> GPU -> float32 / 255 -> ImageNet mean-std",
        "cache_vs_train_max_abs_diff": production_diff,
        "cpu_reference_max_abs_diff": cpu_diff,
        "cpu_reference_ulp": cpu_diff / max(ulp, 1e-12),
        "cpu_reference_fp16_mismatch_elements": half_mismatch,
        "cpu_reference_feature_max_abs_diff": cpu_feat_diff,
        "cross_batch_max_abs_diff": cross,
        "tolerance": {"production": 0.0},
        "pass": bool(production_diff == 0.0 and cpu_diff < 1e-6),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "normalization_check.json").write_text(json.dumps(out, indent=2) + "\n",
                                                     encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
