# 阶段 5+6：通信 / E2E 时延分析与三种 10-MEC 部署配置

生成时间: 2026-11-05
部署: 阶段 2 均匀部署的 10 MEC × 28 张真实地图（`mec_positions_k10.json`）
任务（审计项 4 修正）: **官方口径 21,019** = 11A Train 18,521 + Val-Seen 2,498；
参考口径 27,326（原始 Train 22,002 + Val-Seen + Val-Unseen，见 `e2e_latency_all_tasks.json`）

## 一、时延模型（严格复用 mec-v8.2 `latency_model.py`，commit `d988a9a`）

```
T_E2E = T_encode + T_uplink + T_inference
T_encode    = RTX 5090 实测 1.680 ms（真实 CityNav 裁剪图 → 49×2048）
T_uplink    = payload_bits / a2g_channel(距离).uplink_rate_bps
              （LOS 概率路径损耗 + Shannon；2GHz / GBS 10m / UAV 80m /
                5MHz / 0.1W / NF 9dB；history 由 MEC 侧缓存不重复上行）
payload     = 49 token × 2048 维 × 16 bit = 1,605,632 bit
              （真实 CityNav 视觉 token 尺度；v8.2 合成环境为 ViT 24×24×384×16）
T_inference = FLOPs / MEC 算力（阶段 4 运行期计数：
              Small 4.5M / Medium 55.1M / Transformer-M 556.0M）
T_downlink  = 动作整数回传，忽略（与 v8.2 同口径，v8.2 total 亦不含下行）
```

**算力档位说明**：主分析统一 0.2 TFLOPS/节点——**人为建模假设**，非硬件实测；
审计修正：历史 10-MEC（Stage-2r）使用 **2.0 TFLOPS / 30MHz**；本报告主分析
采用更保守的 0.2 TFLOPS/5MHz 自设档位，算力扫描已覆盖 2.0 TFLOPS 历史值。

## 二、通信时延（official 21,019 任务 × 最近 MEC 距离）

| Mean | P50 | P95 | P99 |
|---|---|---|---|
| **26.42 ms** | 24.67 ms | **36.30 ms** | 46.82 ms |

（最近 MEC 距离均值 65m / P95 142m；1.6 Mbit payload 在 5MHz 信道上占主导）

## 三、三种配置 E2E 时延（@0.2 TFLOPS/节点）

| 配置 | 模型组合 | 推理 | 通信 | 编码 | **E2E Mean** | E2E P95 | E2E P99 | 推理占比 | 通信占比 | 编码占比 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 10×Small | 0.023 ms | 26.42 | 1.68 | **28.1 ms** | 38.0 | 48.5 | 0.1% | 93.8% | 6.1% |
| B | 10×Transformer-M | 2.780 ms | 26.42 | 1.68 | **30.9 ms** | 40.8 | 51.3 | 9.2% | 85.3% | 5.5% |
| C | 5×Small + 5×TM | 1.567 ms | 26.42 | 1.68 | **29.7 ms** | 40.0 | 51.2 | 5.1% | 89.1% | 5.8% |

## 四、重点分析（简报 §5 要求）

1. **Small 与 Transformer-M 的推理时延差**：**2.758 ms**，占 E2E 的 **9.09%**。
2. **Medium 与 Transformer-M 的推理时延差**：**2.504 ms**，占 E2E 的 **8.25%**。
3. **通信是否掩盖模型差异**：**是，通信占绝对主导**。通信均值 26.4ms 是 TM 推理
   （2.78ms）的 **9.5 倍**、是 Small 推理（0.023ms）的 **1150 倍**；编码 1.68ms
   也大于 Small/Medium 的推理。任何模型组合下的 E2E 差异都主要来自通信距离抖动
   （P95−P50 = 11.3ms），而非模型算力。
4. **是否值得用 RL 动态选择 MEC**：
   - 若 RL 目标 = **时延**：收益极小（模型间全差 2.76ms，占 E2E 9%；而同配置内
     仅靠距离变化就有 ±11ms 抖动）。**单纯以算力差异为信号的 RL 训练意义有限**。
   - 若 RL 目标 = **成功率**：模型质量差异巨大（闭环 SR：Small 34.7% vs
     Transformer-M 53.6%，差 18.9 个百分点），且 E2E 时延差异在 100ms 决策预算
     内可接受（TM E2E 30.9ms ≈ 预算 31%）。**RL 应围绕"成功质量 vs 时延"联合
     优化，而非仅时延**——这正是简报要求先做基线测量的价值。
   - 结论：**值得进入 RL 训练，但奖励必须同时包含任务成功与耗时，且必须
     以实测成功率差异（而非仅 FLOPs）作为模型质量信号**；若只训练"最低时延
     选路"，在通信主导下几乎学不到有意义的策略。

## 五、算力敏感性（TM 推理占比随算力）

| 算力 | TM 推理 | TM E2E Mean | 推理占比 |
|---|---|---|---|
| 0.1 TFLOPS | 5.56 ms | 33.6 ms | 16.8% |
| **0.2 TFLOPS** | **2.78 ms** | **30.9 ms** | **9.2%** |
| 0.3 TFLOPS | 1.85 ms | 29.9 ms | 6.3% |
| 0.5 TFLOPS | 1.11 ms | 29.2 ms | 3.9% |
| 1.0 TFLOPS | 0.56 ms | 28.6 ms | 2.0% |

即：即使在最低档 0.1 TFLOPS，推理占比也仅 16.8%，**通信始终主导**。
历史实验若要产生明显的 Small/TM 区分度，必须有更低的算力档（<0.05 TFLOPS）
或更小的 payload；当前真实输入规模不支持"推理主导 E2E"的设定。

## 六、三种部署配置（阶段 6，`deployment_configs.json`）

- **A: 10×Small**、**B: 10×Transformer-M**、**C: 5×Small + 5×Transformer-M**
  （C 按节点索引奇偶交替）
- 三配置**同位置**（阶段 2 均匀部署）、**同带宽 5MHz**、**同发射功率 0.1W**、
  **同信道模型**、**同覆盖半径 500m**、**同任务集**；仅模型类型不同
  （主分析算力同为 0.2 TFLOPS 以隔离模型效应）
- 每节点字段：`map / gbs_id / x / y / model_type / compute_capacity_flops_per_s /
  bandwidth_hz / tx_power_w / coverage_radius_m`
- 文件：`results/deployment_configs.json`（28 图 × 10 节点 × 3 配置）

## 七、图表

- `fig_e2e_breakdown.png`：A/B/C 三段堆叠（编码/通信/推理）
- `fig_share.png`：推理占比 vs 算力（三模型）

## 证据文件

- 脚本：`scripts/analyze_e2e.py`
- 结果：`results/e2e_latency.json`、`results/deployment_configs.json`
- 输入：阶段 2 `mec_positions_k10.json`；阶段 3 `inference_latency.json`；
  阶段 4 `flops_capacity.json`
