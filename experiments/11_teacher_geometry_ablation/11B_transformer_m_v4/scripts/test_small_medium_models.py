#!/usr/bin/env python3
"""Small/Medium 四动作模型单元测试（Task 11）。

Small = MeanPoolMLP（SmallMEC4，V8.2 late-fusion MLP 架构）；
Medium = GoalCrossAttentionMEC（MediumMEC4，goal→current 交叉注意力 + late fusion）；
均适配修正后 V5 的完整多模态输入（含 landmark_geometry 修正几何、masked history 池化）。

CPU 纯 tensor 可跑。运行：python test_small_medium_models.py（退出码 0 = 全部 PASS）
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from test_v5fix_geometry import make_batch  # noqa: E402
from model11_small import SmallMEC4  # noqa: E402
from model11_medium import MediumMEC4  # noqa: E402
from train_11b import MODEL_NAMES, RUN_DIRS, resolve_run_dir  # noqa: E402

TESTS: list[tuple[str, callable]] = []


def test(fn):
    TESTS.append((fn.__name__, fn))
    return fn


def make_lm_batch(b: int) -> tuple[dict, torch.Tensor]:
    lm = torch.zeros(b, 8, 3)
    lm[0, 0, 2] = 0.6
    lm[0, 0, :2] = torch.tensor([10.0, 0.0])
    st = torch.zeros(b, 3)
    return make_batch(b, lm, st), lm


@test
def test_small_forward_shape():
    torch.manual_seed(0)
    m = SmallMEC4().eval()
    batch, _ = make_lm_batch(3)
    with torch.no_grad():
        out = m(batch)
    assert out.shape == (3, 4)
    assert torch.isfinite(out).all()


@test
def test_small_masked_history_pooling():
    torch.manual_seed(1)
    m = SmallMEC4().eval()
    batch, _ = make_lm_batch(2)
    # 两批 mask 相同（槽 10+ 无效），仅被 mask 槽的内容不同（0 vs 垃圾值）
    batch["history_mask"][0, 10:] = 0.0
    batch["history_vis"][0, 10:] = 0.0
    batch["history_pose"][0, 10:] = 0.0
    batch["history_action"][0, 10:] = 0.0
    batch2 = {k: v.clone() for k, v in batch.items()}
    batch2["history_vis"][0, 10:] = 999.0
    batch2["history_pose"][0, 10:] = 999.0
    with torch.no_grad():
        a = m(batch)
        b_out = m(batch2)
    assert torch.equal(a, b_out), "masked history 槽的内容不得影响输出"


@test
def test_small_inactive_landmark_invariance():
    torch.manual_seed(2)
    m = SmallMEC4().eval()
    batch, lm = make_lm_batch(2)
    lm[1, 1, :2] = torch.tensor([999.0, -999.0])
    lm[1, 1, 2] = 0.0
    with torch.no_grad():
        a = m(batch)
    lm[1, 1, :2] = torch.tensor([-500.0, 300.0])
    with torch.no_grad():
        b_out = m(batch)
    assert torch.equal(a, b_out), "无效 landmark 槽（修正几何）不得影响输出"


@test
def test_medium_forward_shape():
    torch.manual_seed(3)
    m = MediumMEC4().eval()
    batch, _ = make_lm_batch(3)
    with torch.no_grad():
        out = m(batch)
    assert out.shape == (3, 4)
    assert torch.isfinite(out).all()


@test
def test_medium_cross_attention_input_sensitivity():
    torch.manual_seed(4)
    m = MediumMEC4().eval()
    batch, _ = make_lm_batch(2)
    with torch.no_grad():
        a = m(batch)
    # 改变 current 视觉输入 → 输出必须变化（goal→current 交叉注意力生效）
    batch2 = {k: v.clone() for k, v in batch.items()}
    batch2["current"][0] += 1.0
    with torch.no_grad():
        b_out = m(batch2)
    assert not torch.equal(a, b_out), "current 变化必须影响输出"
    # 改变 target → 输出必须变化
    batch3 = {k: v.clone() for k, v in batch.items()}
    batch3["target_emb"][0] += 1.0
    with torch.no_grad():
        c = m(batch3)
    assert not torch.equal(a, c), "target 变化必须影响输出"


@test
def test_medium_inactive_landmark_invariance():
    torch.manual_seed(5)
    m = MediumMEC4().eval()
    batch, lm = make_lm_batch(2)
    lm[1, 1, :2] = torch.tensor([999.0, -999.0])
    lm[1, 1, 2] = 0.0
    with torch.no_grad():
        a = m(batch)
    lm[1, 1, :2] = torch.tensor([-500.0, 300.0])
    with torch.no_grad():
        b_out = m(batch)
    assert torch.equal(a, b_out)


@test
def test_run_dir_and_names():
    assert MODEL_NAMES["small"] == "SmallMEC4"
    assert MODEL_NAMES["medium"] == "MediumMEC4"
    assert resolve_run_dir("small", 42, False).name == "seed42"
    assert "runs_small" in str(resolve_run_dir("small", 42, False))
    assert "runs_medium" in str(resolve_run_dir("medium", 42, False))
    assert RUN_DIRS["small"] == "runs_small"
    assert RUN_DIRS["medium"] == "runs_medium"


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
