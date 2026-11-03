#!/usr/bin/env python3
"""V5 修正训练管线单元测试（Task 4）：空候选 blank 合成（修正 5a）与 run 目录保护。

CPU 纯 numpy/torch 可跑，无 CITYNAV_DATA / GPU 依赖。
运行：python test_v5fix_training_plumbing.py（退出码 0 = 全部 PASS）
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from v5fix_contract import BLANK_FRAC, EMPTY_BELIEF, EMPTY_LANDMARKS, EMPTY_MATCH  # noqa: E402
from dataset11b import ChunkedMemmapDataset, apply_blank  # noqa: E402
from train_11b import (build_blank_mask, collate11, resolve_run_dir,  # noqa: E402
                       to_model_batch11)

TESTS: list[tuple[str, callable]] = []


def test(fn):
    TESTS.append((fn.__name__, fn))
    return fn


def fake_sample() -> dict:
    return {"image": np.zeros((3, 3), np.uint8),
            "target_emb": np.ones(512, np.float32),
            "belief": np.arange(10, dtype=np.float32),
            "match": np.ones(5, np.float32),
            "history_vis": np.ones((16, 2048), np.float32),
            "history_pose": np.ones((16, 3), np.float32),
            "history_action": np.eye(4, dtype=np.float32)[0],
            "history_mask": np.ones(16, np.float32),
            "states": np.ones(3, np.float32),
            "landmarks": np.ones((8, 3), np.float32),
            "label": 1}


@test
def test_blanking_passthrough():
    s = fake_sample()
    out = apply_blank(s, False)
    assert out is s, "非 blank 必须原样透传"
    out = apply_blank(s, True)
    assert np.array_equal(out["belief"], EMPTY_BELIEF)
    assert np.array_equal(out["match"], EMPTY_MATCH)
    assert np.array_equal(out["landmarks"], EMPTY_LANDMARKS)
    for k in ("image", "target_emb", "history_vis", "history_pose",
              "history_action", "history_mask", "states", "label"):
        assert np.array_equal(out[k], s[k]), f"{k} 必须透传"
    # 确实替换了（blank 为原地修改：替换后的值 ≠ 原始值）
    s2 = fake_sample()
    orig_belief = s2["belief"].copy()
    out2 = apply_blank(s2, True)
    assert np.array_equal(out2["belief"], EMPTY_BELIEF)
    assert not np.array_equal(out2["belief"], orig_belief), "blank 必须替换 belief"


@test
def test_blank_mask_deterministic_fraction():
    n_ep = 200
    states_per = 5
    ep_order = np.arange(n_ep)
    ep_ranges = {i: (i * states_per, (i + 1) * states_per) for i in range(n_ep)}
    m1 = build_blank_mask(42, 0, ep_order, ep_ranges, BLANK_FRAC)
    m2 = build_blank_mask(42, 0, ep_order, ep_ranges, BLANK_FRAC)
    assert np.array_equal(m1, m2), "同 (seed, epoch) 必须完全复现"
    m3 = build_blank_mask(42, 1, ep_order, ep_ranges, BLANK_FRAC)
    assert not np.array_equal(m1, m3), "不同 epoch 必须轮换"
    assert len(m1) == n_ep * states_per, "与 idx 同长"
    for i in range(n_ep):
        assert (m1[i * states_per:(i + 1) * states_per] == m1[i * states_per]).all(), \
            "每 episode 内恒定"
    n_blank_ep = sum(bool(m1[i * states_per]) for i in range(n_ep))
    p = n_blank_ep / n_ep
    sigma = (BLANK_FRAC * (1 - BLANK_FRAC) / n_ep) ** 0.5
    assert abs(p - BLANK_FRAC) <= 2 * sigma + 1e-9, f"比例 {p} 超出 ±2σ"


@test
def test_run_dir_protection():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        d = resolve_run_dir("v5fix", 42, False, results_root=base)
        assert d == base / "runs_11c_v5fix" / "seed42"
        assert resolve_run_dir("v5", 42, False, results_root=base) \
            == base / "runs_11c" / "seed42"
        assert resolve_run_dir("v4", 42, False, results_root=base) \
            == base / "runs_11b" / "seed42"
        # 已有 best.pt 且非 resume → 拒绝覆盖
        d.mkdir(parents=True)
        (d / "best.pt").write_bytes(b"x")
        try:
            resolve_run_dir("v5fix", 42, False, results_root=base)
            raise AssertionError("应拒绝覆盖旧权重")
        except SystemExit:
            pass
        # --resume 允许
        assert resolve_run_dir("v5fix", 42, True, results_root=base) == d


class _FakeMem:
    """模拟 memmap 数组集（current 物化快路径测试用）。"""

    def __init__(self, with_current: bool):
        self.a = {
            "target_emb": np.zeros((4, 512), np.float32),
            "belief": np.ones((4, 10), np.float32),
            "match": np.ones((4, 5), np.float32),
            "hist_vis": np.ones((4, 16, 2048), np.float16),
            "hist_pose": np.ones((4, 16, 3), np.float32),
            "hist_act": np.zeros((4, 16), np.uint8),
            "hist_mask": np.ones((4, 16), np.float32),
            "states": np.ones((4, 3), np.float32),
            "landmarks": np.ones((4, 8, 3), np.float32),
            "labels": np.ones(4, np.uint8),
            "maps": np.zeros(4, np.uint8),
        }
        if with_current:
            self.a["current"] = np.full((4, 49, 2048), 0.5, np.float16)

    def __getitem__(self, k):
        return self.a[k]


class _RecordingEncoder:
    def __init__(self):
        self.calls = 0
        self.seen = None

    def __call__(self, images):
        self.calls += 1
        self.seen = images
        return torch.zeros(len(images), 49, 2048)


@test
def test_collate_current_passthrough():
    samples = [{"current": np.full((49, 2048), float(i + 1), np.float32),
                "target_emb": np.zeros(512, np.float32),
                "history_vis": np.zeros((16, 2048), np.float32),
                "history_pose": np.zeros((16, 3), np.float32),
                "history_action": np.zeros((16, 4), np.float32),
                "history_mask": np.zeros(16, np.float32),
                "belief": np.zeros(10, np.float32),
                "match": np.zeros(5, np.float32),
                "states": np.zeros(3, np.float32),
                "landmarks": np.zeros((8, 3), np.float32),
                "label": i} for i in range(3)]
    batch = collate11(samples)
    assert "current" in batch and "image" not in batch
    assert batch["current"].shape == (3, 49, 2048)


@test
def test_to_model_batch11_current_fastpath():
    enc = _RecordingEncoder()
    batch_cur = {"current": np.zeros((2, 49, 2048), np.float32),
                 "target_emb": np.zeros((2, 512), np.float32),
                 "history_vis": np.zeros((2, 16, 2048), np.float32),
                 "history_pose": np.zeros((2, 16, 3), np.float32),
                 "history_action": np.zeros((2, 16, 4), np.float32),
                 "history_mask": np.zeros((2, 16), np.float32),
                 "belief": np.zeros((2, 10), np.float32),
                 "match": np.zeros((2, 5), np.float32),
                 "states": np.zeros((2, 3), np.float32),
                 "landmarks": np.zeros((2, 8, 3), np.float32)}
    out = to_model_batch11(batch_cur, enc, "cpu")
    assert enc.calls == 0, "current 快路径不得调用编码器"
    assert out["current"].shape == (2, 49, 2048)
    batch_img = {k: v for k, v in batch_cur.items() if k != "current"}
    batch_img["image"] = np.zeros((2, 7, 7, 3), np.uint8)
    out2 = to_model_batch11(batch_img, enc, "cpu")
    assert enc.calls == 1 and enc.seen.shape == (2, 7, 7, 3), \
        "无 current 时必须走编码器路径"


@test
def test_dataset_current_fallback():
    # 有 current.npy：直接读 memmap，不触 RasterStore（rasters 保持 None）
    ds = ChunkedMemmapDataset.__new__(ChunkedMemmapDataset)
    ds.z = _FakeMem(with_current=True)
    ds.map_names = {}
    ds.idx = np.array([0, 2])
    ds.blank_mask = None
    ds.rasters = None
    ds.has_current = True
    s = ds.__getitem__(0)
    assert "current" in s and "image" not in s
    assert s["current"].shape == (49, 2048) and s["current"].dtype == np.float32
    assert ds.rasters is None, "current 快路径不得创建 RasterStore"
    # 无 current.npy：回退到 image crop 路径
    ds2 = ChunkedMemmapDataset.__new__(ChunkedMemmapDataset)
    ds2.z = _FakeMem(with_current=False)
    ds2.map_names = {0: "m"}
    ds2.idx = np.array([0])
    ds2.blank_mask = None
    ds2.rasters = None
    ds2.has_current = False

    class _Raster:
        def get(self, name):
            return self

        def heading_crop(self, x, y, yaw):
            return np.zeros((7, 7, 3), np.uint8)

    import dataset11b as d11b
    s2 = None
    try:
        s2 = ds2.__getitem__(0)
    except Exception:  # RasterStore 依赖数据，用猴子补丁代替验证
        pass
    if s2 is None:
        ds2.rasters = _Raster()
        s2 = ds2.__getitem__(0)
    assert "image" in s2 and "current" not in s2
    assert s2["image"].shape == (7, 7, 3)


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
