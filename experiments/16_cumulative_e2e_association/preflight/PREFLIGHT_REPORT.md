# Stage 16-P0 · 五模型最终 FLOPs 与 10 TFLOPS 理论时延核验

日期：2026-11-21
前置：冻结 Stage 14 五种最优专家，不重训、不微调。

---

## 1. 审计方法与口径

| 项目 | 说明 |
|---|---|
| 模型来源 | `experiments/14_mec_expert_retraining` 的最终 best.pt（含 stability_revision 的 TS/TM/Large） |
| 参数量来源 | 实际 `sum(p.numel())` |
| FLOPs 统计 | 所有 `nn.Linear` 与 `nn.MultiheadAttention` 的 forward hook |
| MAC→FLOPs | **1 MAC = 2 FLOPs**（与 Stage 13 一致） |
| 统计范围 | 完整 forward + 四分类动作头；视觉 49 token、历史 16 token、注意力、FFN 全部纳入 |
| 未统计 | LayerNorm、GELU、Softmax 等逐元素算子（GFLOPs 量级下可忽略） |
| 不计入 | FrozenEncoder（ResNet50 属 UAV 侧编码，单独计入 encoding latency） |
| 输入 | Batch=1；current (1,49,2048)、target (1,512)、history (1,16,·)、belief (1,10)、match (1,5)、landmarks (1,8,3)、states (1,3) |
| 加速路径 | 将 `TransformerEncoderLayer` 的 fast path 替换为 Python 慢实现，确保逐模块 hook 触发 |

---

## 2. 最终 FLOPs 审计结果

| 模型 | 参数量 | 实测 FLOPs | Stage 13-R2 预测 | 相对误差 | 10 TFLOPS 推理时延 |
|---|---:|---:|---:|---:|---:|
| Small | 61,121,028 | **4.95404 G** | 4.95404 G | +0.000% | **0.495 ms** |
| Medium | 166,011,908 | **14.97853 G** | 14.98062 G | −0.014% | **1.498 ms** |
| Transformer-S | 211,181,060 | **39.67113 G** | 39.67113 G | +0.000% | **3.967 ms** |
| Transformer-M | 498,567,684 | **100.02542 G** | 100.02542 G | +0.000% | **10.003 ms** |
| Large | 1,034,877,444 | **199.72398 G** | 199.72398 G | +0.000% | **19.972 ms** |

结论：实测值与 Stage 13-R2 设计目标一致（最大误差 0.014%），可直接作为 Stage 16
独立时延配置。Stage 12 的旧 `MODEL_GFLOPS`（5/15/40/100/200）仅用于历史对照，
Stage 16 使用本表实测值。

Checkpoint SHA256 与逐模块分解见：
- `model_flops_audit.csv`
- `model_flops_audit.json`
- `module_flops_breakdown.csv`

---

## 3. 部署表（沿用 Stage 12 的五模型均匀部署）

- 10 个 MEC 节点 / 地图；
- 每个节点 10 TFLOPS；
- 模型顺序：Small, Medium, Transformer-S, Transformer-M, Large，重复两次；
- 覆盖半径 500 m；
- 节点坐标来自 `mec_deployment_feasibility/results/mec_positions_k10.json`。

| node_id | model | GFLOPs | 推理时延 @10 TFLOPS |
|---:|---|---:|---:|
| 1 | Small | 4.95404 | 0.495 ms |
| 2 | Medium | 14.97853 | 1.498 ms |
| 3 | Transformer-S | 39.67113 | 3.967 ms |
| 4 | Transformer-M | 100.02542 | 10.003 ms |
| 5 | Large | 199.72398 | 19.972 ms |
| 6 | Small | 4.95404 | 0.495 ms |
| 7 | Medium | 14.97853 | 1.498 ms |
| 8 | Transformer-S | 39.67113 | 3.967 ms |
| 9 | Transformer-M | 100.02542 | 10.003 ms |
| 10 | Large | 199.72398 | 19.972 ms |

完整部署表（含每个地图的具体坐标）见 `deployment_table.csv`。

---

## 4. E2E 时延组成与示例

单步 E2E：

```
T_e2e(t) = T_encode + T_comm(distance) + T_infer(model)
```

| 组件 | 数值 |
|---|---|
| Encoding（ResNet50, 8.174 GFLOPs / 1.5 TFLOPS） | **5.450 ms** |
| Communication（payload 1,605,632 bit，5 MHz，500 m 上限） | 50 m: 24.343 ms；100 m: 28.338 ms；200 m: 50.045 ms；300 m: 74.743 ms；400 m: 99.325 ms；500 m: 125.618 ms |
| Inference（实测 FLOPs / 10 TFLOPS） | Small 0.495 / Medium 1.498 / TS 3.967 / TM 10.003 / Large 19.972 ms |

**示例（距离 100 m）**：

| 模型 | encode | comm | infer | 单步 E2E |
|---|---:|---:|---:|---:|
| Small | 5.450 | 28.338 | 0.495 | **34.28 ms** |
| Medium | 5.450 | 28.338 | 1.498 | **35.29 ms** |
| Transformer-S | 5.450 | 28.338 | 3.967 | **37.75 ms** |
| Transformer-M | 5.450 | 28.338 | 10.003 | **43.79 ms** |
| Large | 5.450 | 28.338 | 19.972 | **53.76 ms** |

**关键观察**：在当前通信模型下，通信时延（24–126 ms）显著大于推理时延
（0.5–20 ms）。因此累计 E2E 主要由**节点距离与通信链路**决定，Min-Latency
策略应优先选择近距离节点，而不是仅看模型 FLOPs。

---

## 5. 实际计时（仅供参考，与理论时延分开）

Stage 12 记录的 RTX 5090 实测（FrozenEncoder）：3.095 ms / call（30 次平均），
理论模型 5.450 ms。Stage 16 的正式时延配置**统一使用理论值**（10 TFLOPS），
不使用实测 GPU 时间，以保证与 Stage 12 的口径可比。

---

## 6. 结论

1. 五模型最终 FLOPs 与 Stage 13-R2 预测一致，无需调整架构。
2. 10 TFLOPS 下推理时延为 0.495–19.972 ms。
3. 通信时延占累计 E2E 的主导部分，Stage 16 的关联策略需要在"节点距离"与
   "模型质量/速度"之间权衡。
4. P0 通过，可进入 P1（环境、累计奖励与终止逻辑）。
