#!/usr/bin/env python3
"""Stage 17 expert pool abstraction (V1 = Stage 16, V2 = reviewed mix).

V2 changes exactly two experts relative to V1:
  small -> Stage-11 SmallMEC4 (2.24 M params, 0.004473344 GFLOPs)
  tm    -> Stage-14-R3 continuation checkpoint (same architecture/FLOPs)

Everything downstream (candidate latency, energy, state features, baselines)
must take its FLOPs map from the same PoolSpec, so weights and latency can
never disagree.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
for p in [HERE, REPO / "experiments/16_cumulative_e2e_association/p1",
          REPO / "experiments/14_mec_expert_retraining",
          REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4/scripts"]:
    sys.path.insert(0, str(p))

KEYS = ["small", "medium", "ts", "tm", "large"]


@dataclass(frozen=True)
class PoolSpec:
    version: str
    models: dict
    flops: dict
    quality: dict
    checkpoints: dict
    sha256: dict


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_pool(version: str = "v2", device: str = "cuda",
              verify_sha: bool = True) -> PoolSpec:
    if version == "v1":
        from env16 import (MODEL_GFLOPS_MEASURED, QUALITY_DEV_F1,  # noqa: E402
                           CKPTS, load_models)
        models = load_models(device)
        return PoolSpec("v1", models, dict(MODEL_GFLOPS_MEASURED),
                        dict(QUALITY_DEV_F1), {k: str(v) for k, v in CKPTS.items()},
                        {})

    cfg = json.loads((ROOT / "configs/expert_pool_v2.json").read_text())
    from models import build_model                       # noqa: E402
    from env16 import NORMALIZED                         # noqa: E402
    models, flops, quality, ckpts, shas = {}, {}, {}, {}, {}
    for key, e in cfg["experts"].items():
        ck = REPO / e["checkpoint"]
        if not ck.exists():
            raise FileNotFoundError(ck)
        if verify_sha and e.get("sha256"):
            got = _sha256(ck)
            if got != e["sha256"]:
                raise RuntimeError(f"{key}: sha mismatch {got} != {e['sha256']}")
            shas[key] = got
        if e["loader"] == "stage11_small_mec4":
            from model11_small import SmallMEC4           # noqa: E402
            m = SmallMEC4(**{k: v for k, v in e.get("kwargs", {}).items()})
        else:
            m = build_model(key)
        sd = torch.load(ck, map_location=device, weights_only=False)
        m.load_state_dict(sd["model"])
        m.to(device).eval()
        for prm in m.parameters():
            prm.requires_grad_(False)
        models[key] = m
        flops[key] = float(e["gflops"])
        quality[key] = e.get("dev_macro_f1")
        ckpts[key] = str(ck)
    return PoolSpec("v2", models, flops, quality, ckpts, shas)


def normalized_models(spec: PoolSpec) -> set:
    """Input standardisation follows the expert, not the pool version."""
    return {"ts", "tm", "large"} if spec.version == "v2" else {"ts", "tm", "large"}


POOL_V2_SUMMARY = {
    "small": {"source": "Stage-11 SmallMEC4", "params": 2_239_236,
              "gflops": 0.004473344},
    "medium": {"source": "Stage-14 Medium", "gflops": 14.978528},
    "ts": {"source": "Stage-14 TS", "gflops": 39.671126},
    "tm": {"source": "Stage-14-R3 TM (epoch 15)", "gflops": 100.025418},
    "large": {"source": "Stage-14 Large (unchanged)", "gflops": 199.723981},
}
