#!/usr/bin/env python3
"""训练/推理契约测试（Task 5/S6）：V5 修正后的输入一致性契约。

全部 CPU / 纯 numpy 可跑（读取已存在的 11B cache npz，无 GPU / CITYNAV_DATA 依赖）。
运行：python test_train_inference_contract.py（退出码 0 = 全部 PASS）
"""
from __future__ import annotations

import math
import sys
import traceback
from collections import deque
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
from continuous_belief import ContinuousBelief  # noqa: E402
from dataset_v2 import history_block  # noqa: E402
from train_11b import collate11  # noqa: E402
from model11c_fix import TransformerM12Fix  # noqa: E402
from dataset11b import apply_blank  # noqa: E402
import v5fix_contract as contract  # noqa: E402
import test_history_pose_consistency as hist_test  # noqa: E402

CACHE = (REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4"
         / "results" / "cache" / "train_seen" / "episodes")
SIGMA_PRIOR = 12.0
TESTS: list[tuple[str, callable]] = []


def test(fn):
    TESTS.append((fn.__name__, fn))
    return fn


def pick_episodes(n: int) -> list[int]:
    all_eps = sorted(int(p.stem) for p in CACHE.glob("*.npz"))
    return [all_eps[i] for i in np.linspace(0, len(all_eps) - 1, n, dtype=int)]


def replay_episode(ep: int):
    """从存储的 match/landmarks 重建 09C 信念演化（无候选库，CPU）。"""
    z = np.load(CACHE / f"{ep}.npz")
    states = z["states"].astype(np.float64)
    match = z["match"]
    landmarks = z["landmarks"]
    T = len(states)

    belief = ContinuousBelief.__new__(ContinuousBelief)
    lm0 = landmarks[0]
    n0 = int((lm0[:, 2] > 0).sum())
    assert n0 == 8, f"ep {ep} t=0 应有 8 个活动分量，got {n0}"
    belief.means = lm0[:n0, :2].astype(np.float64).copy()
    belief.vars = np.full(n0, SIGMA_PRIOR ** 2)
    belief.weights = lm0[:n0, 2].astype(np.float64).copy()
    belief.last_update = -99
    belief.last_accept = None

    for t in range(T):
        x, y, yaw = (float(v) for v in states[t])
        dx, dy, score = (float(v) for v in match[t, :3])
        z_meas = np.array([x + dx, y + dy])
        agree = (belief.last_accept is not None
                 and float(np.linalg.norm(z_meas - belief.last_accept))
                 <= rcl.AGREE_M)
        conf_ok = score >= rcl.TAU
        accept = conf_ok and agree
        if conf_ok:
            belief.last_accept = z_meas.copy()
        if accept:
            belief.update(z_meas, rcl.CAL["sigma_vis_m"],
                          rcl.BEL["best"]["eta"], rcl.BEL["best"]["rho"], t)
        yield t, belief, x, y, yaw, dx, dy, score
    yield T, belief, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # 终止哨兵，便于 zip 检查


@test
def test_history_fields_identical():
    for ep in pick_episodes(3):
        assert hist_test.test_episode(ep), f"episode {ep} history 不一致"


@test
def test_belief_replay_matches_cache():
    for ep in pick_episodes(3):
        z = np.load(CACHE / f"{ep}.npz")
        belief_feat = z["belief"]
        landmarks = z["landmarks"]
        for t, belief, x, y, yaw, dx, dy, score in replay_episode(ep):
            if t >= len(landmarks):
                break
            n_k = int(belief.means.shape[0])
            assert np.allclose(belief.means, landmarks[t, :n_k, :2], atol=1e-4), \
                f"ep {ep} t {t} means 漂移"
            assert np.allclose(belief.weights, landmarks[t, :n_k, 2], atol=1e-6), \
                f"ep {ep} t {t} weights 漂移"
            est = belief.estimate()
            vec = est - np.asarray([x, y])
            fwd = float(vec[0] * math.cos(yaw) + vec[1] * math.sin(yaw))
            lat = float(-vec[0] * math.sin(yaw) + vec[1] * math.cos(yaw))
            dist = float(np.linalg.norm(vec))
            bearing = math.atan2(lat, fwd)
            unc = float(math.sqrt(np.mean(belief.vars)))
            since = float(t - belief.last_update)
            ent = float(belief.entropy())
            if not math.isfinite(ent):
                ent = 0.0
            expect = [fwd, lat, dist, bearing, score, unc, ent, since, dx, dy]
            assert np.allclose(belief_feat[t], expect, atol=1e-4), \
                f"ep {ep} t {t} belief_feat 漂移"


@test
def test_inference_step_inputs_match_cache():
    for ep in pick_episodes(3):
        z = np.load(CACHE / f"{ep}.npz")
        belief_feat = z["belief"]
        match = z["match"]
        landmarks = z["landmarks"]
        for t, belief, x, y, yaw, dx, dy, score in replay_episode(ep):
            if t >= len(landmarks):
                break
            pos = np.asarray([x, y], np.float64)
            est = belief.estimate()
            unc = float(math.sqrt(np.mean(belief.vars)))
            since = float(t - belief.last_update)
            ent = float(belief.entropy())
            if not math.isfinite(ent):
                ent = 0.0
            margin = float(match[t, 3])
            sharp = float(match[t, 4])
            bf = rcl.belief_features(est, pos, yaw, score, unc, ent, since, dx, dy)
            assert np.allclose(bf, belief_feat[t], atol=1e-4), \
                f"ep {ep} t {t} 推理侧 belief 构造漂移"
            mf = np.asarray([dx, dy, score, margin, sharp], np.float32)
            assert np.allclose(mf, match[t], atol=1e-4)
            lm = rcl.landmark_inputs(belief, empty=False)[0]
            n_k = int(belief.means.shape[0])
            assert np.allclose(lm[:n_k, :2], landmarks[t, :n_k, :2], atol=1e-4)
            assert np.allclose(lm[:n_k, 2], landmarks[t, :n_k, 2], atol=1e-6)
            assert not lm[n_k:, :].any(), "空槽必须为 0"


@test
def test_empty_mode_train_inference_contract():
    fake = {"belief": np.ones(10, np.float32), "match": np.ones(5, np.float32),
            "landmarks": np.ones((8, 3), np.float32)}
    out = apply_blank(fake, True)
    assert np.array_equal(out["belief"], rcl.EMPTY_BELIEF), \
        "训练 blank 与推理空候选 belief 必须逐值一致"
    assert np.array_equal(out["match"], rcl.EMPTY_MATCH)
    assert np.array_equal(out["landmarks"], rcl.EMPTY_LANDMARKS)
    lm = rcl.landmark_inputs(None, empty=True)[0]
    assert not lm.any(), "推理空候选 landmarks 必须全 0"


@test
def test_model_batch_dict_contract():
    samples = [{"image": np.zeros((7, 7, 3), np.uint8),
                "target_emb": np.zeros(512, np.float32),
                "history_vis": np.zeros((16, 2048), np.float32),
                "history_pose": np.zeros((16, 3), np.float32),
                "history_action": np.zeros((16, 4), np.float32),
                "history_mask": np.zeros(16, np.float32),
                "belief": np.zeros(10, np.float32),
                "match": np.zeros(5, np.float32),
                "states": np.zeros(3, np.float32),
                "landmarks": np.zeros((8, 3), np.float32),
                "label": 1} for _ in range(3)]
    batch = collate11(samples)
    expect_keys = {"image", "target_emb", "history_vis", "history_pose",
                   "history_action", "history_mask", "belief", "match",
                   "states", "landmarks", "label"}
    assert set(batch) == expect_keys, f"collate 键漂移: {set(batch) ^ expect_keys}"
    # 模型前向冒烟：collate 输出 + encoder 输出占位 → TransformerM12Fix 接受
    torch.manual_seed(5)
    model_batch = {"current": torch.randn(3, 49, 2048)}
    for k in ("target_emb", "history_vis", "history_pose", "history_action",
              "belief", "match", "states", "landmarks", "history_mask"):
        model_batch[k] = torch.from_numpy(batch[k])
    with torch.no_grad():
        out = TransformerM12Fix()(model_batch)
    assert out.shape == (3, 4)


@test
def test_pose_wrap_boundary():
    # pyaw - yaw = π → 两路径必须一致且 wrap 到 -π（修正 6 契约固化）
    pooled = np.zeros((2, 2048), np.float32)
    states = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, math.pi]], np.float64)
    actions = np.array([0, 0], np.int64)
    ref = history_block(pooled, states, actions, 1)
    assert abs(ref["history_pose"][0, 2] - (-math.pi)) < 1e-6, \
        f"history_block π 边界: {ref['history_pose'][0, 2]}"
    vis, pose, act, hmask = rcl.history_inputs(
        deque([pooled[0]]), deque([(0.0, 0.0, 0.0)]), deque([0]),
        np.asarray([1.0, 1.0], np.float64), math.pi)
    assert abs(pose[0, 0, 2] - (-math.pi)) < 1e-6, \
        f"history_inputs π 边界: {pose[0, 0, 2]}"
    assert np.allclose(pose[0], ref["history_pose"], atol=1e-5)


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
