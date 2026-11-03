#!/usr/bin/env python3
"""V5 修正评测侧单元测试（Task 4/S3）：单次编码管线（修正 5b）与空候选常量接线。

CPU 可跑。运行：python test_v5fix_eval_plumbing.py（退出码 0 = 全部 PASS）
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in [HERE,
          REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4/scripts",
          REPO / "experiments/05_transformer_m_training/scripts"]:
    sys.path.insert(0, str(p))

import run_closed_loop as rcl  # noqa: E402
import v5fix_contract as contract  # noqa: E402

TESTS: list[tuple[str, callable]] = []


def test(fn):
    TESTS.append((fn.__name__, fn))
    return fn


@test
def test_single_encode_plumbing():
    """pool_history_from_current 与旧 spatial_maps 路径（NHWC mean+L2 归一）数值一致。"""
    torch.manual_seed(3)
    cur = torch.randn(2, 49, 2048)
    pooled = rcl.pool_history_from_current(cur)
    ref = cur.numpy().reshape(2, 7, 7, 2048).mean(axis=(1, 2))
    ref = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-9)
    assert np.array_equal(pooled, ref), "单次编码的池化必须与旧路径 bit-identical"


@test
def test_empty_mode_constants_wired():
    """run_closed_loop 空候选分支与训练侧共享 v5fix_contract 常量（修正 5a 契约）。"""
    assert rcl.EMPTY_BELIEF is contract.EMPTY_BELIEF, "必须 import 同一常量对象"
    assert np.array_equal(rcl.EMPTY_BELIEF, [0, 0, 0, 0, 0, 50, 0, 99, 0, 0])
    assert not rcl.EMPTY_MATCH.any() and rcl.EMPTY_MATCH.shape == (5,)
    assert not rcl.EMPTY_LANDMARKS.any() and rcl.EMPTY_LANDMARKS.shape == (8, 3)


@test
def test_model_selection_all():
    """build_model 覆盖全部模型名（v4/v5/v5fix/v3/small/medium），未知名拒绝。"""
    from model11b import TransformerM11
    from model11c import TransformerM12
    from model11c_fix import TransformerM12Fix
    from model10c import TransformerM
    from model11_small import SmallMEC4
    from model11_medium import MediumMEC4
    expect = {"v4": TransformerM11, "v5": TransformerM12,
              "v5fix": TransformerM12Fix, "v3": TransformerM,
              "small": SmallMEC4, "medium": MediumMEC4}
    for name, cls in expect.items():
        m = rcl.build_model(name, "cpu")
        assert isinstance(m, cls), f"{name} 应构造 {cls.__name__}"
        assert not any(p.requires_grad for p in m.parameters()), \
            f"{name} 必须是 eval 模式"
    try:
        rcl.build_model("nope", "cpu")
        raise AssertionError("未知模型名必须拒绝")
    except SystemExit:
        pass


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
