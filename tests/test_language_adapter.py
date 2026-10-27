"""Text adapter tests, including the frozen pretrained BERT contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import TextConfig
from src.data.language_adapter import build_language_adapter


def test_hashing_adapter_is_deterministic_and_normalized() -> None:
    adapter, dim, backend, warning = build_language_adapter(
        TextConfig(backend="hashing", hashing_dim=64)
    )
    assert backend == "hashing"
    assert warning is None
    first = adapter.encode_tensor(["go to the red building", "turn left"])
    second = adapter.encode_tensor(["go to the red building", "turn left"])
    assert first.shape == (2, 64)
    assert torch.allclose(first, second)
    assert torch.allclose(
        first.norm(dim=-1), torch.ones(2), atol=1e-6
    )


def test_local_bert_is_loaded_frozen() -> None:
    try:
        adapter, dim, backend, _warning = build_language_adapter(
            TextConfig(backend="bert", local_files_only=True),
            device="cpu",
        )
    except Exception as exc:  # noqa: BLE001 - absent local cache is a skip
        pytest.skip(f"local bert-base-uncased is unavailable: {type(exc).__name__}")
    assert backend == "bert"
    parameters = list(adapter.model.parameters())
    assert parameters
    assert all(not parameter.requires_grad for parameter in parameters)
    before = sum(parameter.detach().double().sum().item() for parameter in parameters)
    features = adapter.encode_tensor(["find the white car", "turn right"])
    after = sum(parameter.detach().double().sum().item() for parameter in parameters)
    assert features.shape == (2, dim)
    assert np.isfinite(features.detach().cpu().numpy()).all()
    assert before == pytest.approx(after, abs=0.0)
    assert all(parameter.grad is None for parameter in parameters)
