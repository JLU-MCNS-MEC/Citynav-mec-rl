# 阶段 3+4：实测推理时延与理论 FLOPs / 算力敏感性报告

生成时间: 2026-11-05
硬件: RTX 5090（32GB，CUDA 13.0）
模型: Small（SmallMEC4，runs_small best）、Medium（MediumMEC4，runs_medium best）、
Transformer-M（TransformerM12Fix = V5fix，runs_11c_v5fix best，当前闭环最好）
输入: val_seen:00000 真实一步，与 run_closed_loop 完全一致的 V5 Fix 多模态契约
（current 49×2048 f32 / target_emb 512 / 16 步历史 / belief / match / Top-8 landmarks / states），
三模型同一 batch、同一设备、同一精度。

## 一、RTX 5090 实测（batch=1，warmup 50 次后 300 次，每次 synchronize）

| 段 | Mean | P50 | P95 | P99 |
|---|---|---|---|---|
| 视觉编码（FrozenEncoder，真实裁剪图 → 49×2048） | 1.680 ms | 1.671 | 1.732 | — |
| 输入预处理（belief/历史/几何装配，CPU） | 0.003 ms | — | — | — |
| **Small 前向** | **2.761 ms** | 2.705 | 3.186 | 3.937 |
| **Medium 前向** | **2.937 ms** | 2.903 | 3.015 | 3.107 |
| **Transformer-M 前向** | **3.355 ms** | 3.348 | 3.420 | 3.460 |

- Small/Transformer-M 实测差：**0.59 ms**（mean）；Medium/Transformer-M 差：**0.42 ms**。
- batch=1 小模型在 5090 上以 GPU 启动开销为主（约 2.7ms 平台），
  FLOPs 差 123× 只反映为 0.6ms —— 真实硬件不能用作 MEC 算力档位的代理。

## 二、单次决策 FLOPs（运行期 hook 计数，2×MACs 口径，与 v8.2 一致）

| 模型 | FLOPs | 相对 Small | 相对 Medium |
|---|---|---|---|
| Small | **4.5 M** | 1× | — |
| Medium | **55.1 M** | 12.2× | 1× |
| Transformer-M | **556.0 M** | **123×** | **10.1×** |

（Small 的 MeanPool 使每个投影只处理 1 个 token；Medium 的 goal→current 交叉注意力
处理 49 个 current token；Transformer-M 的 4 层 × 104 token 编码器占主导。）

## 三、理论时延 T_inf = FLOPs / 算力（算力为**人为建模假设**，非硬件实测）

| 算力 | Small | Medium | Transformer-M | Δ Small→TM | Δ Medium→TM | 理论/实测比值(S/M/TM) |
|---|---|---|---|---|---|---|
| 0.1 TFLOPS | 0.04 ms | 0.55 ms | 5.56 ms | 5.51 ms | 5.01 ms | 0.02 / 0.19 / 1.66 |
| **0.2 TFLOPS** | 0.02 ms | 0.28 ms | **2.78 ms** | **2.76 ms** | **2.50 ms** | 0.01 / 0.09 / 0.83 |
| 0.3 TFLOPS | 0.01 ms | 0.18 ms | 1.85 ms | 1.84 ms | 1.67 ms | 0.01 / 0.06 / 0.55 |
| 0.5 TFLOPS | 0.01 ms | 0.11 ms | 1.11 ms | 1.10 ms | 1.00 ms | 0.00 / 0.04 / 0.33 |
| 1.0 TFLOPS | 0.00 ms | 0.06 ms | 0.56 ms | 0.55 ms | 0.50 ms | 0.00 / 0.02 / 0.17 |

**分析**：
1. 任何档位下 Small→TM 的时延差都接近 Medium→TM 的差（FLOPs 决定比值固定，
   只随 1/算力 缩放）——历史实验若有"Small/TM 显著区分度"必然来自更低的算力
   或更大的 FLOPs 差；**当前真实输入规模下 0.2 TFLOPS 给出 TM≈2.8ms 的推理
   时延与 ~2.5-2.8ms 的模型间差**，作为建模档位合理，但历史出处不可追溯
   （GitHub 仓库单提交、无 Stage-2k/2l 报告），明确标注为建模假设。
2. 0.2 TFLOPS 下 TM 理论值 2.78ms 与 RTX 5090 实测 3.36ms 的比值为 0.83，
   说明 0.2 TFLOPS 建模算力略高于真实 GPU 对小模型的等效吞吐（batch=1
   launch-bound），可用于保守侧建模。
3. 图表：`fig_theoretical_latency.png`、`fig_measured_vs_theoretical.png`。

## 证据文件

- 实测脚本/结果：`scripts/measure_inference_latency.py`、`results/inference_latency.json`
- FLOPs/敏感性脚本/结果：`scripts/estimate_flops_capacity.py`、`results/flops_capacity.json`
- 权重：`runs_small|runs_medium|runs_11c_v5fix/seed42/best.pt`（SHA256 见
  `11B_transformer_m_v4/results/SHA256_MANIFEST.md`）
