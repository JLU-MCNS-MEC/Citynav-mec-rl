#!/usr/bin/env python3
"""Transformer-S / Large（四动作版）单元测试（简报第 2 阶段，TDD RED→GREEN）。

两个模型 = MEC-v8.2 `ExplicitReasoningTransformer` 原始架构（gbs_model_v3.py），
只适配四动作协议的多模态输入投影与融合：
- Transformer-S: hidden_dim=192, num_layers=1, num_heads=4, ff_mult=3
- Large:        hidden_dim=352, num_layers=2, num_heads=8, ff_mult=4

CPU 纯 tensor 可跑。运行：python test_explicit_models.py（退出码 0 = 全部 PASS）
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "experiments/v4_closed_loop_revalidation/scripts"))

from test_v5fix_geometry import make_batch  # noqa: E402
from model11_explicit import TransformerS4, TransformerL4  # noqa: E402
from train_11b import MODEL_NAMES, RUN_DIRS, resolve_run_dir  # noqa: E402
from v5fix_contract import EMPTY_BELIEF, EMPTY_MATCH, EMPTY_LANDMARKS  # noqa: E402

TESTS: list[tuple[str, callable]] = []


def test(fn):
    TESTS.append((fn.__name__, fn))
    return fn


def make_lm_batch(b: int) -> dict:
    lm = torch.zeros(b, 8, 3)
    lm[0, 0, 2] = 0.6
    lm[0, 0, :2] = torch.tensor([10.0, 0.0])
    st = torch.zeros(b, 3)
    return make_batch(b, lm, st)


@test
def test_ts_dims_and_forward():
    torch.manual_seed(0)
    m = TransformerS4().eval()
    assert m.hidden_dim == 192
    assert m.num_layers == 1
    assert m.num_heads == 4
    assert m.ff_mult == 3
    batch = make_lm_batch(3)
    with torch.no_grad():
        out = m(batch)
    assert out.shape == (3, 4)
    assert torch.isfinite(out).all()


@test
def test_large_dims_and_forward():
    torch.manual_seed(1)
    m = TransformerL4().eval()
    assert m.hidden_dim == 352
    assert m.num_layers == 2
    assert m.num_heads == 8
    assert m.ff_mult == 4
    batch = make_lm_batch(3)
    with torch.no_grad():
        out = m(batch)
    assert out.shape == (3, 4)
    assert torch.isfinite(out).all()


@test
def test_fusion_encoder_dims():
    for cls, h, nl, nh, ff in [(TransformerS4, 192, 1, 4, 3),
                               (TransformerL4, 352, 2, 8, 4)]:
        m = cls()
        assert m.fusion_encoder.num_layers == nl
        layer = m.fusion_encoder.layers[0]
        assert layer.self_attn.num_heads == nh
        assert layer.linear1.out_features == h * ff
        assert layer.linear2.in_features == h * ff
        assert m.visual_proj.in_features == 2048
        assert m.visual_proj.out_features == h


@test
def test_goal_current_attention_input_sensitivity():
    torch.manual_seed(2)
    m = TransformerS4().eval()
    assert m.goal_current_attn.num_heads == 4
    batch = make_lm_batch(2)
    with torch.no_grad():
        a = m(batch)
    b2 = {k: v.clone() for k, v in batch.items()}
    b2["current"][0] += 1.0
    with torch.no_grad():
        b = m(b2)
    assert not torch.equal(a, b), "current 变化必须影响输出"
    b3 = {k: v.clone() for k, v in batch.items()}
    b3["target_emb"][0] += 1.0
    with torch.no_grad():
        c = m(b3)
    assert not torch.equal(a, c), "target 变化必须影响输出"


@test
def test_inactive_landmark_invariance():
    torch.manual_seed(3)
    m = TransformerL4().eval()
    batch = make_lm_batch(2)
    batch["landmarks"][1, 1, :2] = torch.tensor([999.0, -999.0])
    batch["landmarks"][1, 1, 2] = 0.0
    with torch.no_grad():
        a = m(batch)
    batch["landmarks"][1, 1, :2] = torch.tensor([-500.0, 300.0])
    with torch.no_grad():
        b = m(batch)
    assert torch.equal(a, b), "无效 landmark 槽（修正几何）不得影响输出"


@test
def test_masked_history_slots_ignored():
    torch.manual_seed(4)
    m = TransformerS4().eval()
    batch = make_lm_batch(2)
    batch["history_mask"][0, 10:] = 0.0
    batch["history_vis"][0, 10:] = 0.0
    batch["history_pose"][0, 10:] = 0.0
    batch["history_action"][0, 10:] = 0.0
    with torch.no_grad():
        a = m(batch)
    batch["history_vis"][0, 10:] = 999.0
    batch["history_pose"][0, 10:] = 999.0
    with torch.no_grad():
        b = m(batch)
    assert torch.equal(a, b), "被 mask 的历史槽内容不得影响输出"


@test
def test_blank_inputs_finite():
    torch.manual_seed(5)
    for cls in (TransformerS4, TransformerL4):
        m = cls().eval()
        batch = make_lm_batch(2)
        batch["belief"][:, :10] = torch.tensor(EMPTY_BELIEF, dtype=torch.float32)
        batch["match"][:, :5] = torch.tensor(EMPTY_MATCH, dtype=torch.float32)
        batch["landmarks"] = torch.zeros(2, 8, 3)
        with torch.no_grad():
            out = m(batch)
        assert torch.isfinite(out).all(), f"{cls.__name__} blank 输入必须有限"


@test
def test_fusion_sequence_length():
    # cls + 49 current + 1 goal + 1 history + 1 state + 8 lm + 28 rel = 89
    torch.manual_seed(6)
    m = TransformerS4().eval()
    captured = {}

    def hook(module, args, output=None):
        captured["seq"] = args[0].shape[1]

    handle = m.fusion_encoder.register_forward_hook(hook)
    with torch.no_grad():
        m(make_lm_batch(2))
    handle.remove()
    assert captured["seq"] == 89, f"fusion 序列应为 89 token，实际 {captured['seq']}"


@test
def test_large_more_params_than_s():
    assert TransformerL4().n_params() > TransformerS4().n_params()


@test
def test_train_registry_and_run_dirs():
    import tempfile
    assert MODEL_NAMES["ts"] == "TransformerS4"
    assert MODEL_NAMES["large"] == "TransformerL4"
    assert RUN_DIRS["ts"] == "runs_transformer_s"
    assert RUN_DIRS["large"] == "runs_large"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assert "runs_transformer_s" in str(
            resolve_run_dir("ts", 42, False, results_root=root))
        assert "runs_large" in str(
            resolve_run_dir("large", 42, False, results_root=root))


@test
def test_run_dir_protection_refuses_overwrite():
    # Small 训练已完成（runs_small/seed42/best.pt 存在）→ 无 --resume 必须拒绝
    try:
        resolve_run_dir("small", 42, False)
    except SystemExit:
        return
    raise AssertionError("已有 best.pt 且无 --resume 时必须拒绝启动")


@test
def test_eval_build_model_factory():
    from run_closed_loop import GEOM_MODELS, build_model  # noqa: E402
    assert "ts" in GEOM_MODELS and "large" in GEOM_MODELS
    assert isinstance(build_model("ts", "cpu"), TransformerS4)
    assert isinstance(build_model("large", "cpu"), TransformerL4)


def main() -> int:
    failures = []
    for name, fn in TESTS:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
            failures.append(name)
    print(f"\n{len(TESTS) - len(failures)}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
