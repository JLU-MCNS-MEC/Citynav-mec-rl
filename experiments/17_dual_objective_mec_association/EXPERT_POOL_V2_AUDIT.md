# Stage 17 · Expert Pool V2 审计

日期：2026-11-28

## 1. 专家配置

| 专家 | 模型 | 权重 | SHA256 | GFLOPs（2×MACs） | Train-Dev Macro-F1 |
|---|---|---|---:|---:|---:|
| **Small** | **Stage-11 SmallMEC4**（2,239,236 参数） | `11B/results/runs_small/seed42/best.pt` | `79b01bd5…d4584d` | **0.004473344** | **0.5937**（新测） |
| Medium | Stage-14 Medium | `checkpoints/Medium/best.pt` | `68f9250f…03bc2e6` | 14.978528 | 0.6025 |
| Transformer-S | Stage-14 TS | `stability_revision/…/ts/…/best.pt` | `9f959616…ef22306b` | 39.671126 | 0.6500 |
| **Transformer-M** | **Stage-14-R3 TM（Epoch 15）** | `tm_r3_lr_continuation/…/best.pt` | `abbfbfe8…d68760d6` | 100.025418 | **0.6407** |
| Large | Stage-14 Large（暂不变） | `stability_revision/…/large/…/best.pt` | `025ede7a…f7742ea` | 199.723981 | 0.6105 |

- 旧 Small 的历史 **0.7243** 来自 Val-Seen 选模，**不能**作为 Train-Dev 先验；
  本轮在 Stage-14 的 Train 内部分割（seed 42, dev_frac 0.02, 25,089 状态）上重新测得
  **0.5937**，并已写入 `configs/expert_pool_v2.json`。
  重新测量不会消除历史 Checkpoint 的 Val-Seen 选模偏差（正式无泄漏对照需重训该架构）。
- 10 TFLOPS 下 Small 推理时延 **0.0004473344 ms**（不再使用 0.4954 ms）。

## 2. 代码接入

| 模块 | 变更 |
|---|---|
| `configs/expert_pool_v2.json` | 五专家的 loader / ckpt / SHA256 / GFLOPs / Dev-F1 |
| `src/pool17.py` | `load_pool("v1"/"v2")` 返回 `PoolSpec(models, flops, quality, ckpts, sha256)`；v2 校验 SHA256，Small 用 `model11_small.SmallMEC4` |
| `src/latency17.py` | `d_inf_ms(model_type, cfg, flops_map=None)`，`service_latency_ms(..., flops_map=)` |
| `src/env17.py` | `EpisodeEnv17(..., flops_map=None)`，候选时延统一走该 map |
| `stage17c/train17c.py`、`evaluate17.py` | 新增 `--pool {v1,v2}`，模型与 FLOPs 同源加载 |

**一致性保证**：模型权重、候选时延、能耗（由时延导出）、状态中的候选时延特征、
三个 Baseline 的 Min-Latency/Max-Quality 选择，全部来自同一 `PoolSpec`，
不存在"权重新、时延旧"的分裂。V1（Stage 16 池）仍可用 `--pool v1` 复现。

## 3. 验证结果（`stage17b/pool_v2_verify.json`，9/9 通过）

| 编号 | 检查 | 结果 |
|---|---|---|
| V1 | 五个模型加载并输出 `[1,4]` | ✅ Small 2,239,236 / Medium 166.0M / TS 211.2M / TM 498.6M / Large 1034.9M |
| V2 | 部署：24 张地图 × 10 节点，5 模型各 2 份，全部 10 TFLOPS | ✅ |
| V3 | 输入张量形状/类型与 Stage 17 batch 一致 | ✅（含 V1 前向） |
| V4 | 旧 Small：V2 加载 vs Stage-11 原类直接构造，logits **max_abs_diff = 0.0** | ✅ |
| V5 | 新 TM：V2 加载 vs `build_model('tm')`+R3 ckpt，**max_abs_diff = 0.0**，ckpt epoch 15 | ✅ |
| V6 | 推理时延：small 0.0004473344 ms、tm 10.0025418 ms，其余不变 | ✅ |
| V7 | 环境 smoke：10 个可行节点，无 `invalid_mec_action` / `no_feasible_mec` | ✅ |
| V8 | 每步专家前向次数 = 1 | ✅ |
| V9 | 每步视觉编码次数 = 1 | ✅ |

**未改动**：10 节点、双份部署、10 TFLOPS、500 m 覆盖、400 ms 截止、100 Wh 预算、
Stage 17 奖励（λ_Q=1.0 / λ_L=0.5 / γ=1.0）、Stage 16 的 `env16.py` 原始定义。
Large-R4 完成后仅需在 `expert_pool_v2.json` 中替换 `large` 条目并更新 SHA。

## 4. 已知限制

1. 旧 Small 的 Checkpoint 仍是"Val-Seen 选模"产物；其 **0.5937** 只是同口径重测，
   不能当作无泄漏的对照数字。
2. Small 的理论时延极小（0.000447 ms），不代表真实 GPU 延迟；按既有理论 FLOPs 模型处理。
3. V2 只在 V1 上替换了两个专家，其余 Three 专家权重不变；因此 V1 与 V2 的 DRL 结果
   **不可直接续训**（旧 P3 Checkpoint 保留为历史实验）。
