"""Anti-leakage tests for the T5 retrieval and inference interfaces."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from src.data.candidates import (
    build_candidate_library,
    build_goal_tokens,
    map_features_from_scores,
)
from src.data.t5_features import history_arrays, history_mask_is_causal
from src.data.trajectory_adapter import Pose5D
from src.models.t5_mec import T5SmallMEC, T5TransformerMMEC
from src.training.retrieval import retrieve_candidates


FORBIDDEN = (
    "target_xyz",
    "target_xy",
    "target_position",
    "object_id",
    "object_ids",
    "ann_id",
    "ann_ids",
    "target_crop",
    "teacher_action",
    "teacher_actions",
    "success",
    "oracle",
)


def _has_forbidden(parameters) -> list[str]:
    names = set(parameters)
    return [name for name in FORBIDDEN if name in names]


def test_retrieval_inference_signature_has_no_oracle_fields() -> None:
    parameters = inspect.signature(retrieve_candidates).parameters
    assert not _has_forbidden(parameters)
    assert list(parameters)[:3] == [
        "language_features", "candidate_embeddings", "retriever"
    ]


def test_candidate_builder_signature_has_no_target_or_object_fields() -> None:
    parameters = inspect.signature(build_candidate_library).parameters
    assert not _has_forbidden(parameters)
    assert "data_config" in parameters
    assert "map_name" in parameters


def test_goal_and_map_feature_builders_do_not_accept_targets() -> None:
    assert not _has_forbidden(inspect.signature(build_goal_tokens).parameters)
    assert not _has_forbidden(inspect.signature(map_features_from_scores).parameters)


def _t5_inputs(batch: int = 2) -> dict:
    return {
        "current_tokens": torch.randn(batch, 576, 384),
        "history_vis": torch.randn(batch, 16, 768),
        "history_pose": torch.randn(batch, 16, 12),
        "history_action": torch.zeros(batch, 16, 6),
        "history_mask": torch.ones(batch, 16),
        "language_features": torch.randn(batch, 768),
        "goal_tokens": torch.randn(batch, 3, 384),
        "gab": torch.randn(batch, 20),
        "map_feat": torch.randn(batch, 22),
    }


@pytest.mark.parametrize("model_cls", [T5SmallMEC, T5TransformerMMEC])
def test_model_forward_rejects_target_coordinate_kwarg(model_cls) -> None:
    model = model_cls()
    inputs = _t5_inputs()
    with pytest.raises(TypeError):
        model.forward_components(**inputs, target_xyz=torch.zeros(2, 3))


@pytest.mark.parametrize("model_cls", [T5SmallMEC, T5TransformerMMEC])
def test_model_forward_rejects_object_id_and_teacher_action(model_cls) -> None:
    model = model_cls()
    inputs = _t5_inputs()
    with pytest.raises(TypeError):
        model.forward_components(**inputs, object_ids=[1, 2])
    with pytest.raises(TypeError):
        model.forward_components(**inputs, teacher_actions=torch.zeros(2, 3))


@pytest.mark.parametrize("model_cls", [T5SmallMEC, T5TransformerMMEC])
def test_map_free_model_rejects_goal_map_and_gab(model_cls) -> None:
    model = model_cls(use_retrieval=False, use_map=False)
    inputs = _t5_inputs()
    baseline = {
        key: inputs[key]
        for key in (
            "current_tokens", "history_vis", "history_pose",
            "history_action", "history_mask", "language_features",
        )
    }
    model.forward_components(**baseline)
    for key in ("goal_tokens", "gab", "map_feat"):
        with pytest.raises(ValueError):
            model.forward_components(
                **baseline, **{key: inputs[key]}
            )


def test_retrieve_candidates_has_no_target_argument() -> None:
    with pytest.raises(TypeError):
        retrieve_candidates(
            torch.zeros(1, 768),
            torch.zeros(4, 384),
            None,
            target_xy=(1.0, 2.0),
        )


def test_history_is_causal_and_masked() -> None:
    poses = [
        Pose5D(0.0, 0.0, 10.0, 0.0, 0.0),
        Pose5D(5.0, 0.0, 10.0, 0.0, 0.0),
        Pose5D(10.0, 0.0, 10.0, 0.0, 0.0),
        Pose5D(15.0, 0.0, 10.0, 0.0, 0.0),
    ]
    actions = [1, 1, 1, 0]
    visuals = np.arange(4 * 768, dtype=np.float32).reshape(4, 768)
    vis, pose, action, mask = history_arrays(poses, actions, 2, visuals)
    assert vis.shape == (16, 768)
    assert pose.shape == (16, 12)
    assert action.shape == (16, 6)
    assert mask[:2].tolist() == [1.0, 1.0]
    assert mask[2:].tolist() == [0.0] * 14
    assert history_mask_is_causal(mask)
    # Slot 0 is the immediate past (state 1), never the current/future state.
    assert np.array_equal(vis[0], visuals[1])
    assert np.array_equal(vis[1], visuals[0])
