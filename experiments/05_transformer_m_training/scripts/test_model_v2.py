"""Contract tests for the stage-05 model and the causal history builder."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

from dataset_v2 import history_block  # noqa: E402
from model_v2 import N_ACTIONS, TransformerM  # noqa: E402

B = 3
KEYS = {"current", "target_emb", "history_vis", "history_pose", "history_action",
        "history_mask", "belief"}


def fake_batch(b: int = B) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "current": torch.randn(b, 7, 7, 2048),
        "target_emb": torch.randn(b, 2048),
        "history_vis": torch.randn(b, 16, 2048),
        "history_pose": torch.randn(b, 16, 3),
        "history_action": torch.zeros(b, 16, 4),
        "history_mask": torch.ones(b, 16),
        "belief": torch.randn(b, 12),
    }


def test_forward_shape_and_single_four_class_head():
    model = TransformerM(d_model=32, n_layers=1, n_heads=4, ffn=64).eval()
    out = model(fake_batch())
    assert out.shape == (B, N_ACTIONS)
    assert model.head.out_features == 4
    # one classifier only: no separate STOP/arrival head anywhere
    heads = [m for m in model.modules() if isinstance(m, torch.nn.Linear)
             and m.out_features == N_ACTIONS]
    assert len(heads) == 1


def test_masked_history_tokens_are_ignored():
    model = TransformerM(d_model=32, n_layers=2, n_heads=4, ffn=64).eval()
    batch = fake_batch()
    batch["history_mask"] = torch.zeros(B, 16)          # no past at all
    with torch.no_grad():
        a = model(batch)
        noisy = {k: (v.clone() if k != "history_mask" else v) for k, v in batch.items()}
        noisy["history_vis"] = torch.randn_like(noisy["history_vis"])
        noisy["history_pose"] = torch.randn_like(noisy["history_pose"])
        noisy["history_action"] = torch.eye(4)[torch.randint(0, 4, (B, 16))]
        b = model(noisy)
    assert torch.allclose(a, b, atol=1e-6)


def test_no_extra_inputs_are_consumed():
    model = TransformerM(d_model=32, n_layers=1, n_heads=4, ffn=64).eval()
    batch = fake_batch()
    assert set(batch) == KEYS
    extra = dict(batch)
    extra["target_xy"] = torch.randn(B, 2)      # must be ignored by design
    with torch.no_grad():
        a = model(batch)
        b = model({k: v for k, v in extra.items() if k in KEYS})
    assert torch.allclose(a, b)


def test_history_is_strictly_causal():
    states = np.asarray([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 2.0, math.pi / 6],
                         [15.0, 2.0, math.pi / 3]], np.float32)
    actions = np.asarray([1, 1, 2, 3], np.int64)
    pooled = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
    t = 3
    hist = history_block(pooled, states, actions, t)
    # slot 0 = state t-1, slot 1 = t-2, ... (most recent first)
    assert np.allclose(hist["history_vis"][0], pooled[2])
    assert np.allclose(hist["history_vis"][1], pooled[1])
    assert np.allclose(hist["history_vis"][2], pooled[0])
    assert hist["history_mask"].tolist() == [1, 1, 1] + [0] * 13
    # action at slot k is the action taken at state t-k-1, never at t
    assert hist["history_action"][0].argmax() == actions[2]
    assert hist["history_action"][1].argmax() == actions[1]
    assert hist["history_action"][0, actions[3]] == 0.0
    # pose is relative to the current state, in the current heading frame
    dx, dy, dyaw = states[2, 0] - states[3, 0], states[2, 1] - states[3, 1], states[2, 2] - states[3, 2]
    c, s = math.cos(states[3, 2]), math.sin(states[3, 2])
    assert np.allclose(hist["history_pose"][0],
                       [dx * c + dy * s, -dx * s + dy * c, (dyaw + math.pi) % (2 * math.pi) - math.pi],
                       atol=1e-5)


def test_history_at_first_state_is_empty():
    states = np.asarray([[1.0, 2.0, 0.3]], np.float32)
    actions = np.asarray([0], np.int64)
    pooled = np.ones((1, 8), np.float32)
    hist = history_block(pooled, states, actions, 0)
    assert hist["history_mask"].sum() == 0
    assert np.allclose(hist["history_vis"], 0.0)


def test_gradients_do_not_reach_the_frozen_encoder():
    torch.manual_seed(0)
    frozen = torch.nn.Linear(2048, 2048)
    frozen.requires_grad_(False)
    model = TransformerM(d_model=32, n_layers=1, n_heads=4, ffn=64)
    batch = fake_batch()
    # the current observation comes from the frozen encoder's output
    with torch.no_grad():
        flat = frozen(torch.randn(B * 49, 2048)).reshape(B, 7, 7, 2048)
    batch["current"] = flat
    loss = model(batch).pow(2).mean()
    loss.backward()
    assert all(p.grad is None for p in frozen.parameters())
    assert any(p.grad is not None for p in model.parameters())


def test_input_dims_match_the_protocol():
    batch = fake_batch()
    assert batch["current"].shape[1:] == (7, 7, 2048)
    assert batch["history_vis"].shape[1:] == (16, 2048)
    assert batch["history_pose"].shape[1:] == (16, 3)
    assert batch["history_action"].shape[1:] == (16, 4)
    assert batch["history_mask"].shape[1:] == (16,)
    assert batch["belief"].shape[1:] == (12,)
