#!/usr/bin/env python3
"""V5 修正（TransformerM12Fix）几何层单元测试 — CPU 纯 tensor，无 CITYNAV_DATA 依赖。

覆盖修正 1-4：
  1. 无效几何 token 的 attention mask（True=pad，与 history 同约定）+ 无效槽特征置零；
  2. 关系角 = heading 帧 bearing 差（旋转不变），dist/50、角/π 归一化；
  3. 28 个 per-pair token [d/50, Δ/π, valid]，无效对 mask+置零，合法合并对 [0,0,1] 可区分；
  4. 槽位按权重降序稳定重排（空槽 w=0 排尾）。

运行：python test_v5fix_geometry.py（退出码 0 = 全部 PASS）
"""
from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from v5fix_contract import (ANGLE_SCALE, DIST_SCALE_M, EMPTY_BELIEF,  # noqa: E402
                            EMPTY_LANDMARKS, EMPTY_MATCH)
from model11c_fix import TransformerM12Fix, N_LM, N_PAIRS, N_REL_FEAT  # noqa: E402
from model11b import N_HISTORY, N_VIS_TOKENS, SPATIAL_CH  # noqa: E402

TESTS: list[tuple[str, callable]] = []


def test(fn):
    TESTS.append((fn.__name__, fn))
    return fn


def make_batch(b: int, landmarks: torch.Tensor, states: torch.Tensor) -> dict:
    """构造与 train_11b.to_model_batch11 同键同形的 CPU batch。"""
    return {
        "current": torch.randn(b, N_VIS_TOKENS, SPATIAL_CH),
        "target_emb": torch.randn(b, 512),
        "history_vis": torch.randn(b, N_HISTORY, SPATIAL_CH),
        "history_pose": torch.randn(b, N_HISTORY, 3),
        "history_action": torch.zeros(b, N_HISTORY, 4),
        "history_mask": torch.ones(b, N_HISTORY),
        "belief": torch.randn(b, 10),
        "match": torch.randn(b, 5),
        "states": states,
        "landmarks": landmarks,
    }


def rot(x, y, th):
    return x * math.cos(th) - y * math.sin(th), x * math.sin(th) + y * math.cos(th)


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


@test
def test_empty_constants():
    assert np.array_equal(EMPTY_BELIEF, [0, 0, 0, 0, 0, 50, 0, 99, 0, 0])
    assert EMPTY_MATCH.shape == (5,) and not EMPTY_MATCH.any()
    assert EMPTY_LANDMARKS.shape == (8, 3) and not EMPTY_LANDMARKS.any()


@test
def test_mask_convention_true_is_pad():
    lm = torch.zeros(1, 8, 3)
    lm[0, 0, 2] = 0.7
    lm[0, 2, 2] = 0.3
    lm[0, 0, :2] = torch.tensor([10.0, 0.0])
    lm[0, 2, :2] = torch.tensor([-5.0, 5.0])
    st = torch.zeros(1, 3)
    _, _, lm_mask, rel_mask = TransformerM12Fix.landmark_geometry(lm, st)
    assert lm_mask.dtype == torch.bool and lm_mask.shape == (1, 8)
    # 修正 4 重排后：槽0=w0.7、槽1=w0.3，其余空槽 w=0 排尾
    expected = torch.tensor([[False, False, True, True, True, True, True, True]])
    assert torch.equal(lm_mask, expected)
    assert rel_mask.dtype == torch.bool and rel_mask.shape == (1, N_PAIRS)
    assert rel_mask[0, 0].item() is False      # 对 (0,1) 有效
    assert rel_mask[0, 1].item() is True       # 对 (0,2) 无效


@test
def test_inactive_slot_features_zeroed():
    # 旧代码对空槽算出指向世界原点的垃圾值（dist=sqrt(100^2+200^2)）；新实现必须置 0
    lm = torch.zeros(1, 8, 3)
    lm[0, 0, 2] = 0.5
    lm[0, 0, :2] = torch.tensor([5.0, 0.0])
    st = torch.tensor([[100.0, 200.0, 0.0]])
    lm_feats, _, _, _ = TransformerM12Fix.landmark_geometry(lm, st)
    for k in range(1, 8):   # 空槽
        assert lm_feats[0, k, 0].item() == 0.0, "dist 必须为 0"
        assert lm_feats[0, k, 1].item() == 0.0, "bearing 必须为 0"
        assert lm_feats[0, k, 2].item() == 0.0


@test
def test_masked_slot_attention_invariance():
    torch.manual_seed(0)
    m = TransformerM12Fix().eval()
    st = torch.zeros(2, 3)
    lm = torch.zeros(2, 8, 3)
    lm[0, 0, 2] = 0.6
    lm[0, 0, :2] = torch.tensor([10.0, 0.0])
    lm[1] = lm[0].clone()
    lm[1, 1, :2] = torch.tensor([999.0, 999.0])   # 被 mask 槽放垃圾坐标
    lm[1, 1, 2] = 0.0
    batch = make_batch(2, lm, st)                  # 其余输入固定，只改被 mask 槽
    with torch.no_grad():
        a = m(batch)
    batch2 = {k: v.clone() for k, v in batch.items()}
    batch2["landmarks"][1, 1, :2] = torch.tensor([-500.0, 300.0])  # 换另一组垃圾
    with torch.no_grad():
        b = m(batch2)
    assert torch.equal(a, b), "被 mask 槽的坐标不应影响输出"


@test
def test_all_lm_rel_masked_forward_finite():
    torch.manual_seed(1)
    m = TransformerM12Fix().eval()
    lm = torch.zeros(2, 8, 3)
    st = torch.zeros(2, 3)
    with torch.no_grad():
        out = m(make_batch(2, lm, st))
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()


@test
def test_bearing_heading_frame_left_positive():
    st = torch.tensor([[0.0, 0.0, 0.0]])
    lm = torch.zeros(1, 8, 3)
    lm[0, 0, :2] = torch.tensor([10.0, 0.0]); lm[0, 0, 2] = 1.0
    lm[0, 1, :2] = torch.tensor([0.0, 10.0]); lm[0, 1, 2] = 1.0
    lm_feats, _, _, _ = TransformerM12Fix.landmark_geometry(lm, st)
    assert abs(lm_feats[0, 0, 1].item() - 0.0) < 1e-6      # 正前
    assert abs(lm_feats[0, 1, 1].item() - 0.5) < 1e-6      # 左 +π/2 → π/2/π = 0.5
    # yaw=π/2（朝 +y），lm 在 (10,0) 处于右侧 → bearing = -π/2
    st2 = torch.tensor([[0.0, 0.0, math.pi / 2]])
    lm2 = torch.zeros(1, 8, 3)
    lm2[0, 0, :2] = torch.tensor([10.0, 0.0]); lm2[0, 0, 2] = 1.0
    lm_feats2, _, _, _ = TransformerM12Fix.landmark_geometry(lm2, st2)
    assert abs(lm_feats2[0, 0, 1].item() - (-0.5)) < 1e-6


@test
def test_rotation_invariance():
    torch.manual_seed(2)
    rng = np.random.default_rng(7)
    st = torch.tensor([[1.0, -2.0, 0.7]])
    lm = torch.zeros(1, 8, 3)
    lm[0, 0, :2] = torch.tensor([20.0, 5.0]); lm[0, 0, 2] = 0.5
    lm[0, 1, :2] = torch.tensor([-10.0, 30.0]); lm[0, 1, 2] = 0.3
    lm[0, 2, :2] = torch.tensor([2.0, 2.0]); lm[0, 2, 2] = 0.15
    ref = TransformerM12Fix.landmark_geometry(lm, st)
    for _ in range(20):
        th = float(rng.uniform(-math.pi, math.pi))
        mx, my = rot(lm[0, :, 0].numpy(), lm[0, :, 1].numpy(), th)
        x2, y2 = rot(st[0, 0].item(), st[0, 1].item(), th)
        st_r = torch.tensor([[x2, y2, wrap(st[0, 2].item() + th)]])
        lm_r = torch.zeros(1, 8, 3)
        lm_r[0, :, 0] = torch.from_numpy(mx)
        lm_r[0, :, 1] = torch.from_numpy(my)
        lm_r[0, :, 2] = lm[0, :, 2]
        got = TransformerM12Fix.landmark_geometry(lm_r, st_r)
        for a, b in zip(ref, got):
            assert torch.allclose(a, b, atol=1e-5), "旋转后特征必须不变"


@test
def test_relation_angle_wrap():
    st = torch.tensor([[0.0, 0.0, 0.0]])
    lm = torch.zeros(1, 8, 3)
    lm[0, 0, :2] = torch.tensor([-9.85, -1.74]); lm[0, 0, 2] = 1.0  # bearing ≈ -170°
    lm[0, 1, :2] = torch.tensor([-9.85, 1.74]); lm[0, 1, 2] = 1.0   # bearing ≈ +170°
    _, rel, _, rel_mask = TransformerM12Fix.landmark_geometry(lm, st)
    # 排序后：槽0=高权重（先）——两权重相等并列，稳定排序保持原序：i=0(-170°), j=1(+170°)
    dang = rel[0, 0, 1].item() * ANGLE_SCALE
    assert abs(dang - (-20 * math.pi / 180)) < 1e-3, f"wrap 应为 -20°, got {dang}"
    assert rel_mask[0, 0].item() is False
    # 边界：bearing_i=0, bearing_j=π → Δ=wrap(π) = -π
    lm2 = torch.zeros(1, 8, 3)
    lm2[0, 0, :2] = torch.tensor([10.0, 0.0]); lm2[0, 0, 2] = 1.0
    lm2[0, 1, :2] = torch.tensor([-10.0, 0.0]); lm2[0, 1, 2] = 0.9
    _, rel2, _, _ = TransformerM12Fix.landmark_geometry(lm2, st)
    assert abs(rel2[0, 0, 1].item() - (-1.0)) < 1e-5, "π 边界必须 wrap 到 -π"


@test
def test_normalization_scales():
    st = torch.tensor([[0.0, 0.0, 0.0]])
    lm = torch.zeros(1, 8, 3)
    lm[0, 0, :2] = torch.tensor([0.0, 50.0]); lm[0, 0, 2] = 0.8  # dist=50, bearing=π/2
    lm_feats, _, _, _ = TransformerM12Fix.landmark_geometry(lm, st)
    assert abs(lm_feats[0, 0, 0].item() - 1.0) < 1e-6    # 50m / 50 = 1.0
    assert abs(lm_feats[0, 0, 1].item() - 0.5) < 1e-6    # π/2 / π = 0.5
    assert abs(lm_feats[0, 0, 2].item() - 0.8) < 1e-6    # w 原值透传


@test
def test_invalid_pair_masked_and_zeroed():
    st = torch.tensor([[0.0, 0.0, 0.0]])
    lm = torch.zeros(1, 8, 3)
    lm[0, 0, :2] = torch.tensor([10.0, 0.0]); lm[0, 0, 2] = 0.6
    lm[0, 1, :2] = torch.tensor([0.0, 10.0]); lm[0, 1, 2] = 0.4
    lm[0, 2, :2] = torch.tensor([50.0, 50.0]); lm[0, 2, 2] = 0.0  # 无效
    lm_feats, rel, lm_mask, rel_mask = TransformerM12Fix.landmark_geometry(lm, st)
    assert rel.shape == (1, N_PAIRS, N_REL_FEAT)
    assert rel_mask[0, 0].item() is False          # (0,1) 有效
    assert rel_mask[0, 1].item() is True           # (0,2) 无效
    assert rel_mask[0, 2].item() is True           # (1,2) 无效
    assert rel[0, 1, 2].item() == 0.0 and rel[0, 1, 0].item() == 0.0
    assert rel[0, 2, 2].item() == 0.0 and rel[0, 2, 0].item() == 0.0
    assert rel[0, 0, 2].item() == 1.0              # 有效对 valid=1
    assert lm_mask[0, 2].item() is True


@test
def test_zero_dist_valid_pair_unmasked():
    st = torch.tensor([[0.0, 0.0, 0.0]])
    lm = torch.zeros(1, 8, 3)
    lm[0, 0, :2] = torch.tensor([5.0, 5.0]); lm[0, 0, 2] = 0.6
    lm[0, 1, :2] = torch.tensor([5.0, 5.0]); lm[0, 1, 2] = 0.4   # 合并对：同一 mean
    _, rel, _, rel_mask = TransformerM12Fix.landmark_geometry(lm, st)
    assert rel_mask[0, 0].item() is False
    assert rel[0, 0, 0].item() == 0.0 and rel[0, 0, 1].item() == 0.0
    assert rel[0, 0, 2].item() == 1.0              # [0,0,1] 与无效对 [0,0,0]+mask 可区分


@test
def test_slot_resort_weight_desc_stable():
    st = torch.tensor([[0.0, 0.0, 0.0]])
    lm = torch.zeros(1, 8, 3)
    w = [0.2, 0.5, 0.2, 0.8]
    xy = [(10.0, 0.0), (0.0, 10.0), (-10.0, 0.0), (0.0, -10.0)]
    for k, (wv, (x, y)) in enumerate(zip(w, xy)):
        lm[0, k, 0] = x; lm[0, k, 1] = y; lm[0, k, 2] = wv
    lm_feats, _, _, _ = TransformerM12Fix.landmark_geometry(lm, st)
    got_w = lm_feats[0, :, 2].tolist()
    assert np.allclose(got_w[:4], [0.8, 0.5, 0.2, 0.2], atol=1e-6), \
        f"降序 + 并列保持原序, got {got_w[:4]}"
    assert all(v == 0.0 for v in got_w[4:]), "空槽 w=0 排尾"


@test
def test_permutation_invariance_active_slots():
    st = torch.tensor([[0.0, 0.0, 0.0]])
    lm = torch.zeros(1, 8, 3)
    data = [(10.0, 0.0, 0.6), (0.0, 10.0, 0.4), (-10.0, 0.0, 0.2)]
    for k, (x, y, w) in enumerate(data):
        lm[0, k, 0] = x; lm[0, k, 1] = y; lm[0, k, 2] = w
    ref = TransformerM12Fix.landmark_geometry(lm, st)
    lm_p = torch.zeros(1, 8, 3)
    for k, idx in enumerate([2, 0, 1]):   # 活动槽任意置换
        lm_p[0, k] = lm[0, idx]
    got = TransformerM12Fix.landmark_geometry(lm_p, st)
    for a, b in zip(ref, got):
        assert torch.allclose(a, b, atol=1e-6), "活动槽置换后特征必须不变"


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
