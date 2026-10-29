"""Contract tests for the fixed-50 m Transformer-M."""

from __future__ import annotations

import torch

from model import N_ACTIONS, TransformerM


def random_batch(b: int = 2) -> dict:
    return {
        "current": torch.randn(b, 7, 7, 2048),
        "target_emb": torch.randn(b, 2048),
        "history_vis": torch.randn(b, 16, 2048),
        "history_pose": torch.randn(b, 16, 3),
        "history_action": torch.randint(0, 4, (b, 16)),
        "history_mask": torch.ones(b, 16),
        "candidate_emb": torch.randn(b, 5, 2048),
        "candidate_meta": torch.randn(b, 5, 4),
        "belief": torch.randn(b, 12),
    }


def test_forward_shapes():
    model = TransformerM(d_model=64, n_layers=1, n_heads=4, ffn=128)
    logits = model(random_batch(3))
    assert logits.shape == (3, N_ACTIONS)
    assert torch.isfinite(logits).all()


def test_single_joint_action_head():
    model = TransformerM(d_model=64, n_layers=1, n_heads=4, ffn=128)
    assert model.head.out_features == N_ACTIONS
    names = [n for n, _ in model.named_modules()]
    assert not any(("arrival" in n) or ("stop_head" in n) for n in names)


def test_no_language_or_bert_modules():
    model = TransformerM(d_model=64, n_layers=1, n_heads=4, ffn=128)
    names = [n.lower() for n, _ in model.named_modules()]
    assert not any(("bert" in n) or ("language" in n) for n in names)


def test_fused_gradients_reach_all_input_branches():
    model = TransformerM(d_model=64, n_layers=1, n_heads=4, ffn=128)
    batch = random_batch(2)
    logits = model(batch)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3]))
    loss.backward()
    for module in (model.vis_proj, model.target_proj, model.hist_vis, model.cand_vis,
                   model.belief_proj[0], model.head):
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all()


def test_masked_history_is_ignored():
    model = TransformerM(d_model=64, n_layers=1, n_heads=4, ffn=128).eval()
    batch = random_batch(2)
    batch["history_mask"] = torch.zeros(2, 16)
    with torch.no_grad():
        a = model(batch)
        batch["history_vis"] = torch.randn(2, 16, 2048)
        b = model(batch)
    assert torch.allclose(a, b, atol=1e-6)


def test_action_embedding_range():
    model = TransformerM(d_model=64, n_layers=1, n_heads=4, ffn=128)
    assert model.hist_action.num_embeddings == 4
