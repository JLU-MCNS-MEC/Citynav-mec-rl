#!/usr/bin/env python3
"""Verify encoder checkpoints: strict load, tensor equality and SHA-256."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.visual_encoder_citynav.model import (
    VisualTokenEncoder,
    encoder_sha256,
    load_pretrained_encoder,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a state dict: {path}")
    return state


def _max_abs_diff(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]):
    if set(left) != set(right):
        return None
    return max(
        (left[key] - right[key]).abs().max().item() for key in left
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or (
        PROJECT_ROOT / "runs" / "visual_encoder_citynav" / args.run_name
    )
    best_path = run_dir / "best_encoder.pt"
    final_path = run_dir / "encoder_final.pt"
    output = args.output or (
        EXPERIMENT_ROOT / "results" / f"{args.run_name}_checkpoint_verification.json"
    )

    best_state = _load_state(best_path)
    final_state = _load_state(final_path)
    encoder, best_load = load_pretrained_encoder(best_path, torch.device("cpu"))
    _, final_load = load_pretrained_encoder(final_path, torch.device("cpu"))
    encoder_keys = set(VisualTokenEncoder().state_dict())

    step_path = run_dir / "checkpoints" / "encoder_step_002751.pt"
    step_check = None
    if step_path.exists():
        step_state = _load_state(step_path)
        step_check = {
            "path": str(step_path),
            "sha256": encoder_sha256(step_path),
            "max_abs_diff_vs_best": _max_abs_diff(step_state, best_state),
        }

    init_path = PROJECT_ROOT / "data_cache" / "weights" / "shared_encoder.pt"
    init_report = None
    param_delta = None
    param_delta_relative = None
    if init_path.exists():
        init_encoder, init_report = load_pretrained_encoder(
            init_path, torch.device("cpu")
        )

        def _vector(module):
            return torch.nn.utils.parameters_to_vector(
                [parameter.detach().float().flatten() for parameter in module.parameters()]
            )

        initial = _vector(init_encoder)
        trained = _vector(encoder)
        param_delta = float((trained - initial).norm().item())
        param_delta_relative = float(param_delta / initial.norm().item())

    auxiliary_keys = sorted(set(best_state) - encoder_keys)
    report = {
        "run_name": args.run_name,
        "best_encoder": {
            "path": str(best_path),
            "sha256": encoder_sha256(best_path),
            "parameters": int(sum(p.numel() for p in encoder.parameters())),
            "strict_load": best_load,
        },
        "final_encoder": {
            "path": str(final_path),
            "sha256": encoder_sha256(final_path),
            "strict_load": final_load,
        },
        "step_002751": step_check,
        "best_vs_final_max_abs_diff": _max_abs_diff(best_state, final_state),
        "best_vs_final_numerically_identical": bool(
            _max_abs_diff(best_state, final_state) == 0.0
        ),
        "auxiliary_head_keys": auxiliary_keys,
        "initialization": init_report,
        "encoder_param_delta_l2": param_delta,
        "encoder_param_delta_relative": param_delta_relative,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"verification written to {output}")
    print(
        "best=%s final=%s identical=%s"
        % (
            report["best_encoder"]["sha256"][:12],
            report["final_encoder"]["sha256"][:12],
            report["best_vs_final_numerically_identical"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
