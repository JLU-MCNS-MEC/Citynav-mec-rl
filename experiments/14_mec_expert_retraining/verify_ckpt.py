#!/usr/bin/env python3
"""Strict Best-Checkpoint reload check before closed-loop evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments/11_teacher_geometry_ablation"
                       "/11B_transformer_m_v4/scripts"))
from models import MODEL_NAMES, build_model  # noqa: E402

EXPECTED = {"small": 61_121_028, "medium": 166_011_908, "ts": 211_181_060,
            "tm": 498_567_684, "large": 1_034_877_444}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(EXPECTED))
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    model = build_model(a.model)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(st["model"], strict=True)   # shape + key check
    n = sum(p.numel() for p in model.parameters())
    bad = [k for k, v in st["model"].items() if not torch.isfinite(v).all()]
    meta = {"model": MODEL_NAMES[a.model], "ckpt": a.ckpt,
            "params": n, "expected_params": EXPECTED[a.model],
            "params_match": n == EXPECTED[a.model],
            "state_keys": len(st["model"]), "nonfinite_tensors": bad,
            "epoch": st.get("epoch"), "macro_f1": st.get("macro_f1")}
    print(json.dumps(meta, indent=1), flush=True)
    ok = n == EXPECTED[a.model] and not bad
    if meta.get("macro_f1") is not None:
        ok &= meta["macro_f1"] > 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
