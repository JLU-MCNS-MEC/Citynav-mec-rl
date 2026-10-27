"""Contract tests for the independent visual-encoder pretraining experiment."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import PROJECT_ROOT
from experiments.visual_encoder_citynav.data import (
    CANDIDATE_K,
    _select_step,
    select_candidate_subset,
)
from experiments.visual_encoder_citynav.losses import multi_positive_infonce
from experiments.visual_encoder_citynav.model import (
    PretrainModel,
    load_pretrained_encoder,
    save_encoder_only,
)


def _model_inputs(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(batch, 3, 336, 336),
        torch.randn(batch, 4, 3, 96, 96),
        torch.randn(batch, 768),
    )


def test_model_forward_has_three_six_action_and_matching_outputs() -> None:
    model = PretrainModel().eval()
    current, candidates, language = _model_inputs()
    with torch.no_grad():
        output = model(current, candidates, language)
    assert output["action_logits"].shape == (2, 6)
    assert output["candidate_tokens"].shape == (2, 4, 384)
    assert output["xview_query"].shape == (2, 256)
    assert output["retrieval_query"].shape == (2, 256)
    assert output["xview_candidate"].shape == (2, 4, 256)
    assert output["retrieval_candidate"].shape == (2, 4, 256)


def test_model_forward_rejects_target_coordinate() -> None:
    model = PretrainModel().eval()
    current, candidates, language = _model_inputs()
    parameters = inspect.signature(model.forward).parameters
    assert "target_xyz" not in parameters
    assert "object_ids" not in parameters
    with pytest.raises(TypeError):
        model(current, candidates, language, target_xyz=torch.zeros(2, 3))


def test_current_and_target_masks_are_distinct() -> None:
    centers = np.asarray(
        [[0.0, 0.0], [10.0, 0.0], [100.0, 0.0], [200.0, 0.0]],
        dtype=np.float32,
    )
    (
        indices, current_mask, target_mask, current_valid, target_valid
    ) = select_candidate_subset(
        centers,
        current_xy=(0.0, 0.0),
        target_xy=(100.0, 0.0),
        rng=np.random.default_rng(0),
        k=CANDIDATE_K,
    )
    assert len(indices) == 4
    assert len(current_mask) == len(target_mask) == CANDIDATE_K
    assert len(current_valid) == len(target_valid) == CANDIDATE_K
    assert not np.array_equal(current_mask, target_mask)
    assert current_mask[0] == 1.0
    assert target_mask[2] == 1.0
    assert current_valid.sum() == 3
    assert target_valid.sum() == 4
    assert np.all(current_valid[len(indices):] == 0)
    assert np.all(target_valid[len(indices):] == 0)


def test_action_step_selection_keeps_actual_label() -> None:
    actions = [1, 2, 3, 0]
    step = _select_step(actions, desired_action=3, index=0)
    assert actions[step] == 3


def test_multi_positive_infonce_is_finite_and_padding_is_excluded() -> None:
    query = torch.randn(2, 8)
    candidates = torch.randn(2, 5, 8)
    positive = torch.tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=torch.float32)
    valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.float32)
    loss, metrics = multi_positive_infonce(
        query, candidates, positive, valid, temperature=0.07
    )
    assert torch.isfinite(loss)
    assert metrics["valid_rows"] == 2
    # A padded candidate must not be selected as the top-1 winner.
    _, metrics_padded = multi_positive_infonce(
        query[:1], candidates[:1], positive[:1], torch.tensor([[1, 1, 0, 0, 0]]),
        temperature=0.07,
    )
    assert metrics_padded["valid_rows"] == 1


def test_encoder_checkpoint_loads_fully_or_fails_explicitly() -> None:
    path = PROJECT_ROOT / "data_cache" / "weights" / "shared_encoder.pt"
    if not path.exists():
        pytest.skip("local shared_encoder.pt is unavailable")
    encoder, report = load_pretrained_encoder(path, torch.device("cpu"))
    assert report["loaded_parameter_ratio"] == pytest.approx(1.0)
    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []
    assert sum(p.numel() for p in encoder.parameters()) == report["total_parameters"]


def test_final_checkpoint_contains_only_visual_encoder(tmp_path: Path) -> None:
    model = PretrainModel()
    path = tmp_path / "encoder.pt"
    report = save_encoder_only(model.encoder, path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert report["parameters"] == sum(p.numel() for p in model.encoder.parameters())
    assert all(key in model.encoder.state_dict() for key in state)
    assert not any(key.startswith("action_head") for key in state)
    assert not any(key.startswith("xview_") for key in state)
    assert not any(key.startswith("retrieval_") for key in state)
