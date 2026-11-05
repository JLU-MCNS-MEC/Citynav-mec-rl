#!/usr/bin/env python3
"""阶段 3：RTX 5090 实测推理时延（Small / Medium / Transformer-M=V5fix）。

- 输入：val_seen:00000 真实一步（birmingham_block_1，真实 heading 裁剪图 +
  真实目标裁剪 + 候选库 belief 初始化），与 run_closed_loop 完全一致的
  V5 Fix 多模态输入契约（current 49×2048 f32 / target_emb 512 / 16 步历史 /
  belief 10 / match 5 / landmarks 8×3 / states 3）。
- 三段计时（同一设备、同一精度、同一 batch）：
  1) 视觉编码：FrozenEncoder（frozen ResNet50）单张 heading 裁剪 → 49×2048；
  2) 输入预处理：目标裁剪→gdesc、候选相似度→belief 初始化、历史/状态装配（CPU）；
  3) 模型前向：三个模型各自 best.pt，eval+no_grad，warmup 50 次后 300 次，
     每次 torch.cuda.synchronize() 计时。
- 输出 results/inference_latency.json（每段 Mean/P50/P95/P99）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
assert (REPO / "experiments").is_dir(), REPO
sys.path.insert(0, str(REPO / "experiments/05_transformer_m_training/scripts"))
sys.path.insert(0, str(REPO / "experiments/11_teacher_geometry_ablation"
                       "/11B_transformer_m_v4/scripts"))
sys.path.insert(0, str(REPO / "experiments/v4_closed_loop_revalidation/scripts"))
sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching/scripts"))
sys.path.insert(0, str(REPO / "experiments/09_online_visual_goal_matching"
                       "/09C_continuous_localization/scripts"))

from dataset_v2 import CandidateStore, FrozenEncoder, RasterStore  # noqa: E402
from model11_small import SmallMEC4  # noqa: E402
from model11_medium import MediumMEC4  # noqa: E402
from model11c_fix import TransformerM12Fix  # noqa: E402
from v5fix_contract import EMPTY_BELIEF, EMPTY_LANDMARKS, EMPTY_MATCH  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
TARGET_HALF_M = 48.0
WARMUP, REPS = 50, 300


def percentile(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q))


def summarize(name: str, arr: np.ndarray) -> dict:
    return {"name": name,
            "mean_ms": float(arr.mean()) * 1e3,
            "p50_ms": percentile(arr, 50) * 1e3,
            "p95_ms": percentile(arr, 95) * 1e3,
            "p99_ms": percentile(arr, 99) * 1e3,
            "n": len(arr)}


def main() -> int:
    device = "cuda"
    torch.manual_seed(0)
    rasters, cands = RasterStore(), CandidateStore()
    name = "birmingham_block_1"
    raster = rasters.get(name)
    pos = np.asarray([33.6427, 561.3063], np.float64)  # val_seen:00000 start
    yaw = float(0.216853)
    target = np.asarray([368.2188, 440.7812], np.float64)

    enc4 = FrozenEncoder(device)
    lib = cands.get(name)

    # ---- 真实一步输入构造（对齐 run_closed_loop.py:285-380）----
    goal_img = __import__("matching", fromlist=["prepare_goal"]).prepare_goal(
        raster, float(target[0]), float(target[1]))
    matcher_enc = __import__("matching", fromlist=["Encoder"]).Encoder(device)
    goal_l2 = matcher_enc.features(goal_img[None], ("layer2",))["layer2"][0]
    gdesc = goal_l2[:, 13:15, 13:15].mean(axis=(1, 2))
    gdesc = gdesc / np.maximum(np.linalg.norm(gdesc, keepdims=True), 1e-8)

    cur_img_np = np.asarray(raster.heading_crop(float(pos[0]), float(pos[1]),
                                                float(yaw)), np.uint8)

    # belief 初始化（候选库嵌入 @ 目标裁剪特征）
    tvec = enc4.embed([raster.north_crop(float(target[0]), float(target[1]),
                                         TARGET_HALF_M)])[0].astype(np.float32)
    sims = lib.embeddings @ tvec
    belief_w = sims / np.maximum(np.linalg.norm(sims, keepdims=True), 1e-8)
    top = np.argsort(belief_w)[-8:]
    lm = np.zeros((1, 8, 3), np.float32)
    lm[0, :len(top), :2] = lib.positions[top].astype(np.float32)
    lm[0, :len(top), 2] = belief_w[top].astype(np.float32)

    batch = {
        "target_emb": torch.from_numpy(gdesc[None].copy()).to(device),
        "history_vis": torch.zeros(1, 16, 2048, device=device),
        "history_pose": torch.zeros(1, 16, 3, device=device),
        "history_action": torch.zeros(1, 16, 4, device=device),
        "history_mask": torch.zeros(1, 16, device=device),
        "belief": torch.from_numpy(np.asarray(EMPTY_BELIEF, np.float32)[None]
                                   .copy()).to(device),
        "match": torch.from_numpy(np.asarray(EMPTY_MATCH, np.float32)[None]
                                  .copy()).to(device),
        "landmarks": torch.from_numpy(lm).to(device),
        "states": torch.from_numpy(np.asarray([[pos[0], pos[1], yaw]],
                                              np.float32)).to(device),
    }

    # ---- 1) 视觉编码计时 ----
    for _ in range(WARMUP):
        with torch.no_grad():
            _ = enc4(np.asarray([cur_img_np]))
    enc_times = []
    for _ in range(REPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            cur = enc4(np.asarray([cur_img_np]))
        torch.cuda.synchronize()
        enc_times.append(time.perf_counter() - t0)
    batch["current"] = cur.detach().clone()

    # ---- 2) 输入预处理计时（CPU：历史/状态/几何装配；不含编码）----
    from continuous_belief import ContinuousBelief  # noqa: E402
    pre_times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        lm2 = np.zeros((1, 8, 3), np.float32)
        lm2[0, :len(top), :2] = lib.positions[top].astype(np.float32)
        lm2[0, :len(top), 2] = belief_w[top].astype(np.float32)
        b = np.asarray(EMPTY_BELIEF, np.float32)[None]
        m = np.asarray(EMPTY_MATCH, np.float32)[None]
        st = np.asarray([[pos[0], pos[1], yaw]], np.float32)
        pre_times.append(time.perf_counter() - t0)

    # ---- 3) 模型前向计时 ----
    models = {
        "small": (SmallMEC4, "runs_small/seed42/best.pt"),
        "medium": (MediumMEC4, "runs_medium/seed42/best.pt"),
        "transformer_m": (TransformerM12Fix, "runs_11c_v5fix/seed42/best.pt"),
    }
    out = {"encode": summarize("visual_encode", np.asarray(enc_times)),
           "preprocess": summarize("input_preprocess", np.asarray(pre_times))}
    for key, (cls, ckpt_rel) in models.items():
        ckpt = (REPO / "experiments/11_teacher_geometry_ablation"
                "/11B_transformer_m_v4/results" / ckpt_rel)
        model = cls().to(device).eval()
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"] if isinstance(sd, dict)
                              and "model" in sd else sd)
        with torch.no_grad():
            for _ in range(WARMUP):
                model({k: v.clone() for k, v in batch.items()})
        times = []
        with torch.no_grad():
            for _ in range(REPS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(batch)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        out[key] = summarize(f"forward_{key}", np.asarray(times))
        print(f"{key}: forward Mean {out[key]['mean_ms']:.3f}ms "
              f"P50 {out[key]['p50_ms']:.3f} P95 {out[key]['p95_ms']:.3f} "
              f"P99 {out[key]['p99_ms']:.3f}")
    print(f"encode: Mean {out['encode']['mean_ms']:.3f}ms "
          f"P50 {out['encode']['p50_ms']:.3f} P95 {out['encode']['p95_ms']:.3f}")
    print(f"preprocess: Mean {out['preprocess']['mean_ms']:.3f}ms")
    (RES / "inference_latency.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("saved results/inference_latency.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
