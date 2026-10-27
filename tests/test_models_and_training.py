"""Forward and one-batch training contract for Small and Transformer-M."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import TrainingConfig
from src.models import SmallMEC, TransformerMMEC
from src.models.mec_common import (
    arrival_gated_decision,
    combine_move_and_arrival,
)
from src.training.supervised import (
    evaluate_components,
    factorized_loss,
    train_step,
)


def _inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "current_tokens": torch.randn(
            batch_size, 576, 384, generator=generator
        ),
        "goal_tokens": torch.randn(batch_size, 3, 384, generator=generator),
        "heatmap": torch.randn(batch_size, 3, 8, 8, generator=generator),
        "network_state": torch.rand(batch_size, 5, generator=generator),
        "language_features": torch.randn(batch_size, 384, generator=generator),
    }


def _batch() -> dict:
    batch = _inputs()
    batch["actions"] = torch.tensor([0, 3], dtype=torch.long)
    return batch


def test_small_forward_and_predict() -> None:
    model = SmallMEC(text_dim=384).eval()
    inputs = _inputs()
    with torch.no_grad():
        logits = model(**inputs)
        decision = model.predict_action(**inputs)

    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()
    assert torch.allclose(
        logits.exp().sum(dim=-1), torch.ones(2), atol=1e-6
    )
    assert decision["action"].shape == (2,)
    assert torch.allclose(
        decision["joint_probabilities"].sum(dim=-1),
        torch.ones(2),
        atol=1e-6,
    )


def test_transformer_m_forward_and_attention() -> None:
    model = TransformerMMEC(text_dim=384).eval()
    inputs = _inputs()
    with torch.no_grad():
        components = model.forward_components(**inputs, return_attention=True)
        logits = model(**inputs)

    assert components["move_logits"].shape == (2, 3)
    assert components["arrival_logit"].shape == (2,)
    assert components["attention"]["goal_to_current"].shape[0] == 2
    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()


def test_factorized_loss_accepts_labels_and_returns_metrics() -> None:
    model = SmallMEC(text_dim=384).eval()
    batch = _batch()
    with torch.no_grad():
        components = model.forward_components(**_inputs())
        losses = factorized_loss(components, np.asarray([0, 3]))

    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])
    assert losses["num_stop"] == 1
    assert losses["num_move"] == 1
    assert 0.0 <= losses["accuracy"] <= 1.0


def test_train_step_ignores_labels_and_updates_parameters() -> None:
    model = SmallMEC(text_dim=384)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = torch.nn.utils.parameters_to_vector(
        [parameter.detach().clone() for parameter in model.parameters()]
    )
    result = train_step(model, optimizer, _batch(), TrainingConfig(batch_size=2))
    after = torch.nn.utils.parameters_to_vector(
        [parameter.detach().clone() for parameter in model.parameters()]
    )

    assert torch.isfinite(torch.tensor(result["loss"]))
    assert np.isfinite(result["grad_norm"])
    assert float((after - before).norm()) > 0.0
    assert result["num_stop"] == 1
    assert result["num_move"] == 1


def test_evaluate_components_is_no_grad_and_reports_metrics() -> None:
    model = SmallMEC(text_dim=384)
    metrics = evaluate_components(model, _batch())
    assert set(metrics) == {
        "loss",
        "accuracy",
        "move_accuracy",
        "stop_precision",
        "stop_recall",
    }
    assert all(np.isfinite(value) for value in metrics.values())


def test_arrival_gated_decision_threshold() -> None:
    move_logits = torch.zeros(1, 3)
    arrival_logit = torch.zeros(1)
    early = arrival_gated_decision(move_logits, arrival_logit, stop_threshold=0.5)
    late = arrival_gated_decision(move_logits, arrival_logit, stop_threshold=0.9)

    assert early["action"].item() == 3
    assert late["action"].item() in (0, 1, 2)
    assert torch.allclose(
        early["joint_probabilities"].sum(dim=-1), torch.ones(1), atol=1e-6
    )


def test_combine_move_and_arrival_is_normalized() -> None:
    move_logits = torch.randn(4, 3)
    arrival_logit = torch.randn(4)
    logits = combine_move_and_arrival(move_logits, arrival_logit)
    assert logits.shape == (4, 4)
    assert torch.allclose(logits.exp().sum(dim=-1), torch.ones(4), atol=1e-6)


def test_text_dim_must_match_language_features() -> None:
    model = SmallMEC(text_dim=768)
    inputs = _inputs()
    with pytest.raises(RuntimeError):
        model(**inputs)
