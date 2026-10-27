"""Candidate retrieval helper tests, including explicit failure padding."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.candidates import (
    build_goal_tokens,
    map_features_from_scores,
    positive_mask_for_target,
    recall_at_k,
    topk_indices,
)
from src.data.trajectory_adapter import Pose5D


def test_goal_tokens_zero_pad_when_fewer_than_top3() -> None:
    embeddings = np.arange(2 * 384, dtype=np.float32).reshape(2, 384)
    tokens, valid = build_goal_tokens(embeddings, [1], k=3)
    assert tokens.shape == (3, 384)
    assert valid == 1
    assert np.array_equal(tokens[0], embeddings[1])
    assert np.all(tokens[1:] == 0)


def test_map_features_zero_pad_when_fewer_than_top5() -> None:
    pose = Pose5D(0.0, 0.0, 50.0, 0.0, 0.0)
    centers = np.asarray([[10.0, 0.0], [0.0, 20.0]], dtype=np.float32)
    scores = np.asarray([0.9, 0.8], dtype=np.float32)
    features, order = map_features_from_scores(pose, centers, scores, top_k=5)
    assert features.shape == (22,)
    assert order.tolist() == [0, 1]
    assert features[-2] == pytest.approx(0.9)
    assert np.all(features[8:20] == 0)  # slots for missing candidates 3-5


def test_recall_at_k_requires_a_positive() -> None:
    scores = np.asarray([0.1, 0.9, 0.5])
    positive = np.asarray([False, True, False])
    result = recall_at_k(scores, positive, ks=(1, 5))
    assert result["recall@1"] is True
    assert result["recall@5"] is True
    no_positive = recall_at_k(scores, np.zeros(3, dtype=bool), ks=(1,))
    assert no_positive["recall@1"] is None


def test_positive_label_helper_is_training_only_by_contract() -> None:
    centers = np.asarray([[0.0, 0.0], [100.0, 100.0]], dtype=np.float32)
    labels = positive_mask_for_target(centers, (0.0, 0.0), radius_m=40.0)
    assert labels.tolist() == [True, False]


def test_topk_indices_are_sorted_by_score() -> None:
    order = topk_indices(np.asarray([0.2, 0.9, 0.1]), 2)
    assert order.tolist() == [1, 0]
