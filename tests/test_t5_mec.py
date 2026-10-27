"""Forward and one-step training tests for the T5 + language models."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import TrainingConfig
from src.models.t5_mec import T5SmallMEC, T5TransformerMMEC
from src.training.supervised import factorized_loss, train_step


def _inputs(batch: int = 2, retrieval: bool = True) -> dict:
    inputs = {
        "current_tokens": torch.randn(batch, 576, 384),
        "history_vis": torch.randn(batch, 16, 768),
        "history_pose": torch.randn(batch, 16, 12),
        "history_action": torch.zeros(batch, 16, 6),
        "history_mask": torch.ones(batch, 16),
        "language_features": torch.randn(batch, 768),
    }
    if retrieval:
        inputs.update(
            {
                "goal_tokens": torch.randn(batch, 3, 384),
                "gab": torch.randn(batch, 20),
                "map_feat": torch.randn(batch, 22),
            }
        )
    return inputs


@pytest.mark.parametrize("model_cls", [T5SmallMEC, T5TransformerMMEC])
def test_full_t5_forward_is_six_action(model_cls) -> None:
    model = model_cls()
    inputs = _inputs(retrieval=True)
    with torch.no_grad():
        logits = model(**inputs)
        decision = model.predict_action(**inputs)
    assert logits.shape == (2, 6)
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits.exp().sum(dim=-1), torch.ones(2), atol=1e-6)
    assert decision["action"].shape == (2,)
    assert torch.allclose(
        decision["joint_probabilities"].sum(dim=-1), torch.ones(2), atol=1e-6
    )


@pytest.mark.parametrize("model_cls", [T5SmallMEC, T5TransformerMMEC])
def test_map_free_t5_forward(model_cls) -> None:
    model = model_cls(use_retrieval=False, use_map=False)
    inputs = _inputs(retrieval=False)
    with torch.no_grad():
        logits = model(**inputs)
    assert logits.shape == (2, 6)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("model_cls", [T5SmallMEC, T5TransformerMMEC])
def test_t5_factorized_loss_and_train_step(model_cls) -> None:
    model = model_cls()
    inputs = _inputs(retrieval=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = torch.nn.utils.parameters_to_vector(
        [p.detach().clone() for p in model.parameters()]
    )
    batch = {**inputs, "actions": torch.tensor([1, 0], dtype=torch.long)}
    with torch.no_grad():
        components = model.forward_components(**inputs)
        losses = factorized_loss(components, batch["actions"])
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])
    result = train_step(model, optimizer, batch, TrainingConfig(batch_size=2))
    after = torch.nn.utils.parameters_to_vector(
        [p.detach().clone() for p in model.parameters()]
    )
    assert np.isfinite(result["loss"])
    assert np.isfinite(result["grad_norm"])
    assert float((after - before).norm()) > 0.0
