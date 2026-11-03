#!/usr/bin/env python3
"""V5 修正契约测试补全（4 个缺口，Task S7）：与 test_train_inference_contract.py 互补。

1. test_old_v5_weights_loadable — 旧 v5 权重只能由冻结旧类 TransformerM12 加载
   （rel_proj 56→3 的结构变化使 v5fix 类拒载，保护"旧权重永不误用"契约）；
2. test_landmark_slot_order_contract — 训练 memmap landmarks 与推理 harness
   landmark_inputs 逐槽一致（含顺序），钉住"memmap 无需重建"契约；
3. test_blanked_sample_model_equiv — 训练 blank 样本与推理空候选批次的模型
   输出逐位一致（模型级契约，补值级 test_empty_mode_train_inference_contract）；
4. test_full_batch_equivalence_memmap_vs_harness — 同一真实状态经训练批处理
   管线（collate11/to_model_batch11）与推理 harness 构造逐字段等价（含 dtype）。

CPU 可跑（读取 11B cache npz + memmap，无 CITYNAV_DATA / GPU 依赖）。
运行：python test_v5fix_contract_gaps.py（退出码 0 = 全部 PASS）
"""
from __future__ import annotations

import json
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
from train_11b import collate11, to_model_batch11  # noqa: E402
from dataset11b import apply_blank, open_memmap  # noqa: E402
from model11c import TransformerM12  # noqa: E402
from model11c_fix import TransformerM12Fix  # noqa: E402
from test_train_inference_contract import (CACHE, pick_episodes,  # noqa: E402
                                           replay_episode)

MEM = (REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4"
       / "results" / "data" / "train_seen_memmap")
OLD_V5_BEST = (REPO / "experiments/11_teacher_geometry_ablation"
               "/11B_transformer_m_v4/results/runs_11c/seed42/best.pt")
SIGMA_PRIOR = 12.0
TESTS: list[tuple[str, callable]] = []


def test(fn):
    TESTS.append((fn.__name__, fn))
    return fn


def load_ep_ranges() -> dict[int, tuple[int, int]]:
    raw = json.loads((MEM / "ep_ranges.json").read_text(encoding="utf-8"))
    return {int(k): tuple(v) for k, v in raw.items()}


class _StubEncoder:
    """替身编码器：固定 current 张量，隔离视觉编码（CPU 契约测试专用）。"""

    def __init__(self, current: torch.Tensor):
        self.current = current

    def __call__(self, images) -> torch.Tensor:
        return self.current


@test
def test_old_v5_weights_loadable():
    ckpt = torch.load(OLD_V5_BEST, map_location="cpu", weights_only=False)
    old = TransformerM12()
    old.load_state_dict(ckpt["model"])          # strict：键必须完全匹配
    with torch.no_grad():
        # 前向冒烟：旧权重 + 旧类可输出 (1,4)
        batch = {"current": torch.randn(1, 49, 2048),
                 "target_emb": torch.randn(1, 512),
                 "history_vis": torch.randn(1, 16, 2048),
                 "history_pose": torch.randn(1, 16, 3),
                 "history_action": torch.randn(1, 16, 4),
                 "history_mask": torch.ones(1, 16),
                 "belief": torch.randn(1, 10),
                 "match": torch.randn(1, 5),
                 "states": torch.randn(1, 3),
                 "landmarks": torch.rand(1, 8, 3)}
        out = old(batch)
    assert out.shape == (1, 4)
    # 结构保护：rel_proj 输入 56→3，v5fix 类必须拒载旧权重（形状不匹配必抛）
    try:
        TransformerM12Fix().load_state_dict(ckpt["model"], strict=False)
        raise AssertionError("v5fix 类不应能加载旧 v5 权重（rel_proj 56→3）")
    except RuntimeError:
        pass


@test
def test_landmark_slot_order_contract():
    ep_ranges = load_ep_ranges()
    lm_mem = np.lib.format.open_memmap(MEM / "landmarks.npy", mode="r")
    for ep in pick_episodes(3):
        s, e = ep_ranges[ep]
        for t, belief, *_ in replay_episode(ep):
            if t >= e - s:
                break
            lm_harness = rcl.landmark_inputs(belief, empty=False)[0]
            lm_memmap = np.asarray(lm_mem[s + t], np.float32)
            n_k = int((lm_memmap[:, 2] > 0).sum())
            assert n_k == int(belief.means.shape[0]), \
                f"ep {ep} t {t} 活动分量数不一致 {n_k} vs {belief.means.shape[0]}"
            # 槽序契约：逐槽比对（回放 float64→float32 有 ~1e-6 精度漂移，
            # 与 test_train_inference_contract 同款 atol；顺序错位会产生远大于
            # 该容差的差异，因此逐槽 allclose 即钉住顺序）
            assert np.allclose(lm_harness[:n_k], lm_memmap[:n_k], atol=1e-4), \
                f"ep {ep} t {t} memmap 与 harness 槽序/数值不一致"
            assert np.array_equal(lm_harness[n_k:], lm_memmap[n_k:]), \
                f"ep {ep} t {t} 尾部空槽非零或计数错位"


@test
def test_blanked_sample_model_equiv():
    torch.manual_seed(7)
    base = {"image": np.zeros((7, 7, 3), np.uint8),
            "target_emb": np.random.randn(512).astype(np.float32),
            "history_vis": np.random.randn(16, 2048).astype(np.float32),
            "history_pose": np.random.randn(16, 3).astype(np.float32),
            "history_action": np.eye(4, dtype=np.float32)[1],
            "history_mask": np.ones(16, np.float32),
            "belief": np.random.randn(10).astype(np.float32),
            "match": np.random.randn(5).astype(np.float32),
            "states": np.asarray([12.3, -45.6, 0.7], np.float32),
            "landmarks": np.random.rand(8, 3).astype(np.float32),
            "label": 1}
    # 训练路径：blank 合成（原地替换 belief/match/landmarks）
    blanked = apply_blank({**base}, True)
    # 推理路径：空候选分支（harness 同源常量构造）
    harness = {**base,
               "belief": rcl.EMPTY_BELIEF.copy(),
               "match": rcl.EMPTY_MATCH.copy(),
               "landmarks": rcl.landmark_inputs(None, empty=True)[0].copy()}
    for k in ("belief", "match", "landmarks"):
        assert np.array_equal(blanked[k], harness[k]), \
            f"{k} 训练 blank 与推理空候选不一致（值级）"
    batch_blank = collate11([blanked])
    batch_harness = collate11([harness])
    cur = torch.randn(1, 49, 2048)
    enc = _StubEncoder(cur)
    mb_blank = to_model_batch11(batch_blank, enc, "cpu")
    mb_harness = to_model_batch11(batch_harness, enc, "cpu")
    for k in ("belief", "match", "states", "landmarks"):
        assert torch.equal(mb_blank[k], mb_harness[k]), \
            f"模型批 {k} 两路径不一致"
    model = TransformerM12Fix().eval()
    with torch.no_grad():
        out_blank = model(mb_blank)
        out_harness = model(mb_harness)
    assert torch.equal(out_blank, out_harness), \
        "blank 训练样本与空候选推理批次的模型输出必须逐位一致"


@test
def test_full_batch_equivalence_memmap_vs_harness():
    ep = pick_episodes(1)[0]
    z = np.load(CACHE / f"{ep}.npz")
    T = len(z["states"])
    t = T // 2
    ep_ranges = load_ep_ranges()
    s, e = ep_ranges[ep]
    k = s + t

    # ── 训练侧：直接读 memmap 数组（不经 RasterStore，CPU 可跑）──
    mem = open_memmap(MEM, ("hist_vis", "hist_pose", "hist_act",
                            "hist_mask", "belief", "match", "target_emb",
                            "states", "labels", "landmarks"))
    train_sample = {"image": np.zeros((7, 7, 3), np.uint8),   # stub 编码器忽略
                    "target_emb": np.asarray(mem["target_emb"][k], np.float32),
                    "belief": np.asarray(mem["belief"][k], np.float32),
                    "match": np.asarray(mem["match"][k], np.float32),
                    "history_vis": mem["hist_vis"][k].astype(np.float32),
                    "history_pose": np.asarray(mem["hist_pose"][k], np.float32),
                    "history_action": np.eye(4, dtype=np.float32)[mem["hist_act"][k]],
                    "history_mask": np.asarray(mem["hist_mask"][k], np.float32),
                    "states": np.asarray(mem["states"][k], np.float32),
                    "landmarks": np.asarray(mem["landmarks"][k], np.float32),
                    "label": int(mem["labels"][k])}

    # ── 推理侧：harness 按真实闭环逐步重建同一状态 ──
    pooled = z["pooled"].astype(np.float32)
    states = z["states"].astype(np.float64)
    actions = z["actions"].astype(np.int64)
    hist_vis: deque = deque(maxlen=rcl.HISTORY_LEN)
    hist_pose: deque = deque(maxlen=rcl.HISTORY_LEN)
    hist_act: deque = deque(maxlen=rcl.HISTORY_LEN)
    for step in range(t):
        hist_vis.append(pooled[step])
        hist_pose.append((float(states[step, 0]), float(states[step, 1]),
                          float(states[step, 2])))
        hist_act.append(int(actions[step]))
    # 信念回放至 t（与 test_train_inference_contract.replay_episode 同源重建）
    belief = None
    for tt, b, *_ in replay_episode(ep):
        if tt == t:
            belief = b
            break
    assert belief is not None
    x, y, yaw = (float(v) for v in states[t])
    dx, dy, score = (float(v) for v in z["match"][t, :3])
    est = belief.estimate()
    unc = float(np.sqrt(np.mean(belief.vars)))
    since = float(t - belief.last_update)
    ent = float(belief.entropy())
    if not np.isfinite(ent):
        ent = 0.0
    vis, pose_h, act_h, hmask = rcl.history_inputs(
        hist_vis, hist_pose, hist_act, states[t, :2], yaw)
    harness_sample = {"image": np.zeros((7, 7, 3), np.uint8),
                      "target_emb": train_sample["target_emb"].copy(),
                      "belief": rcl.belief_features(est, np.asarray([x, y]),
                                                    yaw, score, unc, ent,
                                                    since, dx, dy),
                      "match": np.asarray(z["match"][t], np.float32),
                      "history_vis": vis[0],
                      "history_pose": pose_h[0],
                      "history_action": act_h[0],
                      "history_mask": hmask[0],
                      "states": np.asarray(states[t], np.float32),
                      "landmarks": rcl.landmark_inputs(belief, empty=False)[0],
                      "label": int(mem["labels"][k])}

    # ── 全批管线对拍（值 + dtype）──
    batch_train = collate11([train_sample])
    batch_harness = collate11([harness_sample])
    cur = torch.randn(1, 49, 2048)
    enc = _StubEncoder(cur)
    mb_train = to_model_batch11(batch_train, enc, "cpu")
    mb_harness = to_model_batch11(batch_harness, enc, "cpu")
    for key in ("target_emb", "history_vis", "history_pose", "history_action",
                "history_mask", "belief", "match", "states", "landmarks"):
        assert mb_train[key].dtype == mb_harness[key].dtype, \
            f"{key} dtype 不一致: {mb_train[key].dtype} vs {mb_harness[key].dtype}"
        # match/states 为 npz 逐字拷贝（float32→float64→float32 精确往返），
        # 其余字段经 float64 回放重建，有 ~1e-6 精度漂移（与既有契约测试同 atol）
        if key in ("match", "states", "history_mask"):
            assert torch.equal(mb_train[key], mb_harness[key]), \
                f"{key} 训练 memmap 与推理 harness 批张量不一致"
        else:
            assert torch.allclose(mb_train[key], mb_harness[key], atol=1e-4), \
                f"{key} 训练 memmap 与推理 harness 批张量不一致"
    model = TransformerM12Fix().eval()
    with torch.no_grad():
        out_train = model(mb_train)
        out_harness = model(mb_harness)
    assert torch.allclose(out_train, out_harness, atol=1e-5), \
        "同一状态经训练/推理两路径的模型输出必须一致（容差 1e-5）"


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
