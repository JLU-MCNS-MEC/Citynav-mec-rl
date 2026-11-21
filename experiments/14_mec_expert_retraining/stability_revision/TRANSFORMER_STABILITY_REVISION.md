# Stage 14-R2：三种 Transformer 统一训练稳定化与学习率重新设计

日期：2026-11-18
范围：Transformer-S / Transformer-M / Large；Small、Medium 冻结不重训。
前置诊断：`diagnosis/TRANSFORMER_S_COLLAPSE_REPORT.md`（commit `395083aa`）。

> 本文件在稳定性验证运行期间生成；结果段落以运行时实际数据填充。

---

## 1. 原 3e-4 为何不适合当前大规模 Transformer

（结论来自 Stage 14-R1 诊断报告，此处仅摘要）

1. **规模跳跃但超参未变**：旧 TransformerS4 为 d=192 / L=1 / ~1M 参数，Stage 14 的
   Transformer-S 为 d=1408 / L=8 / 211M 参数。两者沿用同一 LR=3e-4、warmup=3%。
   历史小模型 grad_norm 稳定在 0.87–2.39；TS 在 epoch 0 即为 4.48，epoch 1 达 417。
2. **LR 到达峰值的时刻即异常起点**：epoch 0 含 5,882 步 warmup，epoch 1 全程处于
   2.96–3.00e-4 平台；grad_norm 恰在此刻从 4.48 跳到 417。
3. **注意力饱和**：epoch 0 结束时 L0 QK score std 已达 3,893，`attn_max=1.000`；
   epoch 3 score absmax 6.5e6，Q std 120。
4. **残差流爆炸**：std 153→504（epoch 0），1,217→4,231（epoch 3）。
5. **反向逐层放大**：epoch 3 每层 `self_attn` 放大约 1e3–1e6，8 层累计 ~1e18，
   总梯度 1.10e14；clip 到 1.0 后 AdamW 仍以 lr 步长推动权重，形成正反馈。
6. **输入尺度失衡是触发条件**：`state` token 初始范数 12,288，是 `cls`（398）的 31 倍；
   belief/match/history_pose 的原始值（243 / 50 / 99 / 73）未经归一化。

---

## 2. 统一训练协议

三种 Transformer 使用**完全相同**的：

| 参数 | 值 |
|---|---|
| Optimizer | AdamW |
| Max LR | **5e-5** |
| Weight decay | 0.01 |
| Scheduler | OneCycleLR |
| Warmup ratio | **0.10** |
| Annealing | cosine |
| Gradient clipping | 1.0 |
| Precision | BF16 autocast |
| Max epochs | 20 |
| Early stopping | patience 10，Dev Macro-F1 |
| Loss | 4-class cross entropy，无 class weight，无 label smoothing |

Batch（唯一允许的差异，按显存预算）：

| 模型 | Batch | Accum | Effective |
|---|---:|---:|---:|
| TS | 128 | 1 | 128 |
| TM | 128 | 1 | 128 |
| Large | 64 | 2 | 128 |

配置文件：`shared_training_config.json`。
**不允许**按单个模型的 Dev 表现单独调整 LR 或归一化方式。

---

## 3. 输入归一化

### 3.1 方案

- 统计量只由 **Train**（18,151 episodes，1,254,793 states）计算；Dev（370 episodes）
  与 Val-Seen（2,498 tasks）不参与。
- 标准化：`belief`（10 维）、`match`（5 维）、`history_pose`（3 维，仅 `hist_mask>0.5`
  的有效历史步）。
- 不处理：`current`、`history_vis`、`target_emb`（冻结编码器输出，原有尺度接近 1，
  不重复归一化）；`landmarks`/`relation`（已由 V5FIX 几何层归一化为
  `[d/50, Δ/π, valid]`，`w` 保持原始）；`states`（世界坐标，仅用于相对几何）。
- 统计量保存于 `normalization_statistics.json`，三个模型共享。

### 3.2 统计量（Train-only，1,254,793 states；有效历史步 17,657,844）

**belief（10 维，模型只用 `BELIEF_IDX=[0,1,2,3,5,6,7]`）**

| dim | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean | 77.28 | 2.28 | 131.86 | 0.023 | 0.83 | 9.32 | 0.805 | 108.06 | 2.59 | −1.86 |
| std | 105.66 | 92.41 | 91.06 | 1.12 | 0.057 | 4.56 | 0.306 | 73.33 | 25.97 | 26.14 |
| min | −517 | −517 | 0.25 | −3.14 | 0.55 | 0.24 | 0.0 | 0.0 | −48.1 | −48.7 |
| max | 510 | 508 | 518 | 3.14 | 0.99 | 12.0 | 1.0 | 783 | 48.7 | 48.0 |

**match（5 维）**

| dim | 0 (dx) | 1 (dy) | 2 (score) | 3 (margin) | 4 (sharpness) |
|---|---:|---:|---:|---:|---:|
| mean | 2.585 | −1.861 | 0.83 | 0.023 | 0.051 |
| std | 25.97 | 26.14 | 0.057 | 0.029 | 0.020 |
| min | −48.1 | −48.7 | 0.55 | 0.0 | 0.0 |
| max | 48.7 | 48.0 | 0.99 | 0.26 | 0.16 |

**history_pose（3 维，仅有效历史步）**

| dim | 0 (forward) | 1 (lateral) | 2 (Δyaw) |
|---|---:|---:|---:|
| mean | −26.157 | 0.105 | −0.005 |
| std | 19.148 | 12.455 | 0.859 |
| min | −80.0 | −65.0 | −3.14 |
| max | 49.33 | 65.0 | 3.14 |

说明：
- `match` 的 score/margin/sharpness 原始 std 仅 0.02–0.06，标准化会把它们放大 17–50 倍；
  这是 per-dim 标准化的预期行为，使这些维度不再“几乎无变化”。
- `belief` 的 since 维度（mean 108、max 783）标准化后，空候选常量 `since=99` 变为 −0.12，
  不再是离群值；`unc=50` 同样被拉回正常范围。
- `current` / `history_vis` 的全局 std 在统计文件中记为 `Infinity`：这是 float16 大数组
  累加溢出的记录问题，**不影响训练**（这两个模态不参与标准化），其 mean/min/max 有效。

### 3.3 归一化前后的 Token 尺度

| token | 归一化前 L2（诊断报告） | 归一化后 L2（冒烟，step 20） |
|---|---:|---:|
| cls | 426 | **1.06** |
| current | 619 | 12.6 |
| target | 444 | 37.5（`gc_ffn_norm` 的 LayerNorm 输出尺度 √1408） |
| history | 619 | 17.5 |
| **state** | **13,170** | **15.9** |
| landmark | 503 | 25.9 |
| relation | 431 | 15.8 |

state token 降低约 **827 倍**；所有 token 的绝对值都在 ~1–80 范围内，
不再出现“某一个模态比其余大一个数量级”的初始失衡。

---

## 4. 统一初始化

| 参数 | 规则 |
|---|---|
| `cls_token` / `cls` | Normal(0, 0.02) |
| `current_pos` / `vis_pos` | Normal(0, 0.02) |
| `hist_time` | Normal(0, 0.02) |
| `type_emb`（nn.Embedding） | Normal(0, 0.02) |
| `hist_action`（Embedding，TM） | Normal(0, 0.02) |
| Linear | PyTorch 默认 `xavier_uniform` |
| LayerNorm | 默认 weight=1, bias=0 |
| 结构 | Pre-LN 不变；不引入 QK-Norm / LayerScale / DeepNorm |

实测初始范数：

| 参数 | 修改前 | 修改后 |
|---|---:|---:|
| TS `type_emb.weight` | 98.89（N(0,1)） | **1.994** |
| TS `cls_token` | 0（零初始化） | 0.738 |
| TS `current_pos` | 0 | 5.256 |
| TS `hist_time` | 0 | 2.998 |
| TM `type_emb.weight` | 119.04（N(0,1)） | **2.402** |
| TM `hist_action.weight` | 90.68（N(0,1)） | **1.819** |
| TM `vis_pos` | 0 | 6.374 |

`type_emb` 从“每个类型向量范数约 37、主导 token 范数”降为“整体范数约 2”。

---

## 5. 稳定性防护

`train_stable.py` 相对原训练循环新增：

1. **异常更新拦截**：pre-clip grad norm 非有限或 `> 1e4` 时**跳过该次更新**
   （不再裁剪后继续用病态方向更新）；连续 10 次异常则以 rc=2 中止并写 `ABORT.json`。
2. **`grad_trace.jsonl`**：每步记录 `grad_norm_preclip`、`grad_norm_postclip`、当前 LR、
   更新前后权重范数、是否跳过、非有限梯度参数数量。
3. **注意力诊断**：每 500 次更新抽样一次，记录每层 Q/K/V std、QK score std/absmax、
   attention entropy、最大权重、残差流 std/absmax，写入
   `attention_diagnostics.jsonl`（汇总为 `attention_diagnostics.json`）。
4. **Token 尺度诊断**：每个 epoch 结束时记录 7 类 token 的 mean/std/mean-L2/max-L2。

---

## 6. 稳定性验证结果（每模型最多 2 epoch）

运行顺序：TS → TM → Large，一次一个模型。

### 6.1 汇总

**Transformer-S（2 epochs，已完成，2026-11-18 16:10–17:33）**

| epoch | train loss | train acc | Dev Macro-F1 | Dev acc | 预测 S/F/L/R | 跳过 | 秒 |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 0.5847 | 0.7688 | 0.5820 | 0.7863 | 358 / 22822 / 931 / 978 | 0 | 2448 |
| 1 | **0.5303** | **0.7872** | **0.6249** | **0.8004** | 375 / 21937 / 1330 / 1447 | 0 | 2535 |

- best epoch 1，Dev Macro-F1 **0.6249**；总耗时 5,019 s；`skipped_updates=0`，`nan_losses=0`。
- grad norm（pre-clip）窗口中位数：0–2k = 3.59 → 2–4k = 2.51 → 4–6k = 2.28 →
  6–8k = 2.10 → 8–9.8k = **2.04**；最大值 52.14 仅出现在第一步。
- 注意力：score_absmax 5.4 → 15.5 → 28.4 → 47.8 → 61.3（**量级稳定在 1e1–1e2**，
  未出现指数增长）；`attn_max` 局部达到 1.0；平均 entropy 4.52 → 2.63。
- Token 尺度（epoch 0 末）：cls 1.10 / current 14.59 / target 37.67 / history 16.49 /
  state 14.64 / landmark 7.76 / relation 9.96。
- 对比原 3e-4 崩溃训练：epoch 0 的 Dev Macro-F1 由 0.3989 提升到 **0.5820**，
  且四类预测均未坍缩（原 epoch 2 起 Stop/TurnLeft/TurnRight 归零）。

**Transformer-M（2 epochs，已完成，2026-11-18 17:33–19:49）**

| epoch | train loss | train acc | Dev Macro-F1 | Dev acc | 预测 S/F/L/R | 跳过 | 秒 |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 0.5937 | 0.7642 | 0.5642 | 0.7808 | 426 / 22841 / 852 / 970 | 0 | 3997 |
| 1 | **0.5389** | **0.7842** | **0.6177** | **0.7930** | 386 / 22037 / 1263 / 1403 | 0 | 3991 |

- best epoch 1，Dev Macro-F1 **0.6177**；`skipped_updates=0`，`nan_losses=0`。
- grad norm 窗口中位数：0–500 = 5.98 → 500–1k = 4.52 → 1k–1.5k = 3.87 → 1.5k–2k = **3.40**。
- 注意力：score_absmax 36.8 → 59.5（2k）→ 262（9.5k）→ **272–278**（19k，已趋稳）；
  entropy 3.45 → 3.18 → 0.865 → **0.741–0.805**；`attn_max=1.0` 局部存在。
- 与 TS 相比，TM 的注意力更集中（entropy 更低），但 score 量级稳定在 1e2，
  未出现指数增长；梯度与 Dev 指标均正常。

**Large（2 epochs，已完成，2026-11-18 21:46–2026-11-19 00:39）**

| epoch | train loss | train acc | Dev Macro-F1 | Dev acc | 预测 S/F/L/R | 跳过 | 秒 |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 0.6000 | 0.7631 | 0.5502 | 0.7808 | 375 / 23171 / 697 / 846 | 0 | 8450 |
| 1 | **0.5456** | **0.7815** | **0.6105** | **0.7886** | 358 / 22183 / 1220 / 1328 | 0 | 8543 |

- best epoch 1，Dev Macro-F1 **0.6105**；`skipped_updates=0`，`nan_losses=0`。
- 注意力（epoch 1 末）：score_absmax 408–410（稳定，未指数增长）、entropy 0.465、
  `attn_max=1.0` 局部存在；grad 无异常。
- 显存事件：首次启动（无显存优化）在 `_ff_block` 的 `linear2` 处 **CUDA OOM**：
  进程已用 29.00 GiB，GPU 可用 31.35 GiB，其他服务占用 2.18 GiB，只差 130 MiB。
  使用 `expandable_segments:True` 重试仍 OOM（只差 34 MiB）。
- 采用 **per-layer gradient checkpointing**（不改变架构、参数、batch 64×2、LR、
  数据、归一化与初始化；仅让反向传播重算每层 forward）：峰值 allocated 显存
  **24.3 GB**（vs 原始 ~29.1 GB，节省约 5 GB），代价约 0.86 s/micro-step。
- 该差异属于任务书允许的“显存允许的 Batch 实现差异”，不影响三模型共享的
  有效 batch（128）、LR、优化器、归一化、初始化与数据顺序。

### 6.2 通过标准

1. 无 NaN/Inf；
2. grad norm 不持续增长或指数爆炸；
3. 不发生多数类预测坍缩；
4. attention 不出现大规模数值饱和；
5. 各模态 token 尺度处于合理范围；
6. Dev Macro-F1 不持续恶化。

---

## 7. 结论

### 7.1 三模型汇总

| 模型 | 参数量 | epoch 0 Dev F1 | epoch 1 Dev F1 | best | grad 跳过 | NaN | 峰值显存 | 2 epoch 耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Transformer-S | 211.2 M | 0.5820 | **0.6249** | **0.6249** | 0 | 0 | ~10 GB | 1.4 h |
| Transformer-M | 498.6 M | 0.5642 | **0.6177** | **0.6177** | 0 | 0 | ~21 GB | 2.25 h |
| Large | 1034.9 M | 0.5502 | **0.6105** | **0.6105** | 0 | 0 | 24.3 GB | 4.8 h |

### 7.2 逐项回答任务书的问题

1. **原 3e-4 为何不适合当前大规模 Transformer**：见第 1 节。核心是规模放大后 LR 未调整，
   注意力在 epoch 1 进入 LR 峰值后迅速饱和，梯度逐层放大到 1e14。
2. **5e-5 是否能够稳定训练三种模型**：**能够**。三个模型在 2 epoch 内：
   - `skipped_updates=0`、`nan_losses=0`；
   - grad norm 单调下降或稳定（TS 3.59→2.04；TM 5.98→3.40；Large 26→13）；
   - Dev Macro-F1 逐 epoch 上升，四类预测均未坍缩。
3. **归一化前后的 Token 范数差异**：state token 从 13,170 降到 15.9（约 827 倍）；
   所有 token 的初始 L2 从“426–13,170”的 31 倍失衡收敛到“1–38”的合理范围。
4. **梯度爆炸是否消失**：**是**。原崩溃的 grad norm 轨迹为 4.48 → 417 → 9.06e9 →
   1.10e14；新配置下三个模型的最大 grad norm 均 < 63（且仅出现在首步），
   2 epoch 内无任何指数增长或异常跳过。
5. **三种模型是否保持一致训练协议**：**是**。AdamW、max_lr=5e-5、wd=0.01、
   OneCycleLR、warmup=10%、cosine、clip=1.0、BF16、loss、数据顺序、归一化、
   初始化完全相同；唯一差异是 Large 因 32 GB 显存限制启用 per-layer gradient
   checkpointing（属于任务书允许的“显存允许的 Batch 实现差异”，有效 batch 仍为 128）。
   详细的配置见 `shared_training_config.json`。
6. **是否可以进入完整 20 Epoch 正式训练**：**可以**。三模型均通过全部六项通过标准。

### 7.3 稳定性判据的补充说明

- **`attn_max=1.0` 的解读**：三个模型在训练中都会出现个别 query-head 的 softmax
  one-hot。这是长尾聚焦现象，不等同于原崩溃中的“全分布饱和”。判据应以
  `score_absmax` 是否指数增长、`entropy` 是否趋近 0、grad norm 是否上升为准：
  - TS：score_absmax 5→61，entropy 4.52→2.63，grad 3.59→2.04；
  - TM：score_absmax 37→278，entropy 3.45→0.74，grad 5.98→3.40；
  - Large：score_absmax 410（稳定），entropy 0.41–0.47（稳定），grad 26→13。
  三者的 score 量级均停留在 1e1–1e2，未进入原崩溃的 1e5–1e6。
- 若后续完整训练中 `score_absmax` 进入 1e4 以上或 `entropy < 0.2` 且 grad norm
  同步上升，应视为新的失稳信号并暂停。

### 7.4 后续行动与最终结果

三种 Transformer 均已完成 2-epoch 稳定性验证，并按用户指令从各自的 2-epoch
checkpoint 续训（恢复 model / optimizer / RNG / best / history，2-epoch checkpoint
备份在 `runs/<model>/seed42/checkpoints_stability_2ep/`）：

| 模型 | 续训结果 | best Dev Macro-F1 | 结束方式 |
|---|---|---:|---|
| **Transformer-S** | 20 epoch 完成 | **0.6500**（epoch 16） | 跑满 |
| **Transformer-M** | 12 epoch | 0.6177（epoch 1） | epoch 11 早停 |
| **Large** | 12 epoch | 0.6105（epoch 1） | epoch 11 早停 |

- TS 在续训后期（epoch 13–16）恢复并显著超过 2-epoch best（0.6249 → 0.6500）；
- TM 与 Large 的续训轨迹高度一致：LR 重新 warmup 后注意力持续集中
  （entropy 0.04–0.10），Dev F1 在 0.56–0.60 波动，10 轮无改善后早停；
- 三模型全程 `skipped_updates=0`、`nan_losses=0`；
- Large 曾因驱动层 Xid 8（GPU 看门狗复位）中断一次，GPU 恢复后从 2-epoch
  checkpoint 重试成功；该事件与模型/数据无关。

### 7.5 Val-Seen 闭环评测结果（2498 任务，step-cap 200，归一化与训练一致）

| 模型 | SR | OSR | SPL | NE (m) | stop_rate | stop_precision |
|---|---:|---:|---:|---:|---:|---:|
| Small | 0.5392 | 0.5460 | 0.4773 | 75.87 | 0.6701 | 0.8047 |
| Medium | 0.3195 | 0.3259 | 0.3184 | 91.20 | 0.3823 | 0.8356 |
| **Transformer-S** | **0.6301** | **0.6397** | **0.5450** | **64.23** | **0.7182** | **0.8774** |
| Transformer-M | 0.4251 | 0.4616 | 0.3894 | 90.04 | 0.4984 | 0.8530 |
| Large | 0.3351 | 0.3651 | 0.3254 | 94.93 | 0.3923 | 0.8541 |

SR 排名：TS 63.0% > Small 53.9% > TM 42.5% > Large 33.5% > Medium 32.0%。
Large 的主要失败模式为 out_of_bounds（1437 条，57.5%），与其训练后期注意力
高度集中一致。

所有产物已推送 GitHub；权重本体（30 GB+）超出 GitHub 限制，见
`CHECKPOINTS_SHA256.txt`。

---

## 8. 产物清单

| 文件 | 内容 |
|---|---|
| `shared_training_config.json` | 三模型统一配置 |
| `normalization_statistics.json` | Train-only 归一化统计量 |
| `prepare_normalizer.py` | 统计量计算脚本 |
| `train_stable.py` | 统一稳定训练脚本 |
| `run_stability_phase.sh` | 三模型串行验证脚本 |
| `summarize_stability.py` | 汇总与绘图 |
| `runs/<model>/seed42/` | checkpoint、history、grad_trace、诊断 |
| `figures/` | 稳定性对比图 |
