# Transformer-S 训练崩溃诊断报告

日期：2026-11-18
对象：Stage 14-R1 `Transformer-S`（`models.py::TransformerPreset(1408, 8, 8, 4224, 5376)`）
范围：只诊断 Transformer-S；未修改训练代码、未删除/覆盖历史权重、未启动任何新训练。

---

## 0. 执行记录（安全停止）

| 时间 | 事件 |
|---|---|
| 15:13:18 | epoch 3 的 `last.pt` 完整落盘（`torch.save` 结束） |
| 15:13:43 | `train_progress.json` 写入 epoch=3（写在该 checkpoint 之后，作为安全停止信号） |
| 15:13:46 | 监控脚本发送 **SIGTERM**（非 kill -9） |
| 15:13:51 | 主进程 PID 3152247 正常退出 |
| 15:13:46 | Supervisor 记录 `ts: train rc=-15` → `ts: TRAIN FAILED, stopping`，**自行退出** |
| 15:14 | 清理 8 个孤儿 DataLoader worker（父进程退出后残留，PPID=1），使用 SIGTERM |

结果：
- `best.pt` 未变（仍为 epoch 0，11:22:37，844,766,297 B）；
- `last.pt` 按预期更新为 **epoch 3**（2,534,321,555 B），epoch 3 的 `last.pt` 覆盖了 epoch 2 的版本；
- **未启动 Transformer-M / Large**，**未执行 Val-Seen 闭环评测**；
- 本次诊断全部为只读操作（加载 checkpoint、前向/反向统计），未写任何 checkpoint。

---

## 1. 结论摘要

**Transformer-S 的失败不是 Early Stopping 问题，而是注意力饱和 → 残差流爆炸 → 梯度逐层放大的正反馈崩溃。**

| epoch | train loss | train acc | grad_norm（clip 前均值） | Dev Macro-F1 | Dev 预测分布 (S/F/L/R) |
|---|---:|---:|---:|---:|---|
| 0 | 0.6486 | 0.7509 | **4.48** | **0.3989** | 272 / 24301 / 486 / 30 |
| 1 | 0.6813 | 0.7452 | **4.17e2** | 0.3171 | 89 / 24324 / 675 / 1 |
| 2 | 0.7734 | 0.7374 | **9.06e9** | 0.2138 | 0 / 25076 / 1 / 12 |
| 3 | 1.1582 | 0.5821 | **1.10e14** | 0.2139 | **0 / 25089 / 0 / 0** |

三层根因：

1. **触发条件 —— 输入尺度失衡 + 超参沿用旧小模型**
   `state` token 初始范数 12,288，是 `cls` token（398）的 **31 倍**；原因是 belief/match/pose 的原始值未经归一化（可达 243 / 50 / 99 / 73）。而 LR=3e-4、warmup=3%、初始化方式全部沿用了 d=192/L=1 时代的小模型配置。
2. **直接机制 —— 注意力完全饱和 + 残差流爆炸**
   epoch 0 时第 0 层 QK score 的 std 已达 3,893，`attn max = 1.000`（softmax 变成 one-hot）；epoch 3 时 score absmax 达 **6.5e6**，Q 的 std 为 **120**（正常应 ≈1）。残差流 std 从 154 增长到 504（epoch 0），再增长到 1,217→4,231（epoch 3）。
3. **放大机制 —— 反向传播逐层放大约 1e3–1e6 倍**
   epoch 3 实测：layer7 的梯度量级约 1e-5，layer0 达到 **1.75e13**；8 层累计放大约 **1e18**。总梯度 1.9e13–8.2e14，与训练日志的 epoch 均值 1.10e14 一致。

**对照证据（同类代码、相同 LR、相同数据）**：历史小模型稳定，说明不是数据/标签/mask/loss 的实现错误。

| run | 结构 | grad_norm 轨迹 |
|---|---|---|
| `runs_transformer_s`（旧 TransformerS4） | d=192, L=1, H=4 | 2.30 → 0.93 → 0.87 → 0.94 → 1.30 → 1.76 → 2.16 → 2.39 |
| `runs_large`（旧 TransformerL4） | d=352, L=2, H=8 | 1.93 → 1.13 → 1.03 → 0.99 → 1.04 → 1.24 → 1.46 → 1.63 |
| `runs_11c_v5fix`（TransformerM12Fix） | d=256, L=4 | 1.43 → 0.88 → 0.78 → 0.72 → 0.67 → 0.63 → 0.59 → 0.57 |
| **Stage 14 TS** | **d=1408, L=8, 211M** | **4.48 → 417 → 9.06e9 → 1.10e14** |

---

## 2. 梯度异常分析

### 2.1 统计口径（逐项核对代码）

`train_expert.py` 的更新块：

```python
gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
if window_ok and math.isfinite(gn):
    optimizer.step(); scheduler.step()
```

| 检查项 | 结论 |
|---|---|
| grad norm 统计在裁剪前还是后 | **裁剪前**。`clip_grad_norm_` 返回的是裁剪发生之前的总范数，日志/`handoff` 中的 9.06e9、1.10e14 都是**裁剪前**值 |
| 裁剪是否正确执行 | 正确。`gn` 有限时执行 `optimizer.step()`；`clip_coef = 1.0/gn`，裁剪后范数恒为 1.0 |
| 返回值是否被正确理解 | 是。代码把它当“裁剪前范数”记录，语义正确；但**日志只记录 epoch 平均值**，无法看到单步尖峰 |
| 梯度累积除法 | `BATCH_CFG["ts"] = (128, 1)`，`accum=1`，`(loss/accum).backward()` 与 `micro % accum == 0` 一致，无漏除 |
| Loss 重复加权 | `F.cross_entropy(logits, label)`，无 class weight、无 label smoothing、无重复乘系数 |
| 重复反向传播 | 每个 micro-batch 恰好一次 `backward()`；`zero_grad(set_to_none=True)` 在每次 `optimizer.step()` 后执行 |
| NaN/Inf | `nan_updates=0`；本次实测 4 个 batch 的梯度 `nonfinite_grad_params=0` |

**防护缺陷（代码问题，非本次根因）**：只在 `gn` 非有限时跳过更新。9.06e9 / 1.10e14 这样的**有限但病态**梯度仍会被裁剪后照常更新，相当于用被破坏的方向继续以 lr 步长推动权重。

### 2.2 实测总梯度（同一数据顺序，4 个 batch）

| | batch 0 | batch 1 | batch 2 | batch 3 |
|---|---:|---:|---:|---:|
| epoch 0 权重（best.pt） | 3.78 | 20.17 | 3.96 | 108.76 |
| epoch 3 权重（last.pt） | 1.92e13 | 8.21e14 | 1.54e14 | 1.40e14 |

裁剪后范数：两者都被压到 1.0。差别在于**方向的有效性**：epoch 3 的梯度已由数值放大主导。

### 2.3 逐模块梯度（RMS over 4 batches）

**epoch 0**

| 模块 | 梯度范数 | grad absmax |
|---|---:|---:|
| `state_proj.0`（Linear 12→1408） | 46.4 | 6.38 |
| `hist_pose_proj.0`（Linear 3→1408） | 17.0 | 3.55 |
| `encoder.layer0.self_attn` | 14.0 | — |
| `encoder.layer0.norm1` | 12.4 | 2.47 |
| `visual_proj` | 12.2 | 0.89 |

**epoch 3**

| 模块 | 梯度范数 | grad absmax |
|---|---:|---:|
| `visual_proj` | **3.92e14** | 1.39e13 |
| `goal_current_attn` | **1.53e14** | 2.14e12 |
| `encoder.layer0.norm1` | 2.75e13 | 3.82e12 |
| `type_emb` | 2.61e13 | 2.70e12 |
| `hist_pose_proj.0` | 2.13e13 | 3.52e12 |
| `state_proj.0` | 1.52e13 | — |
| `lm_proj.0` | 1.32e13 | 2.20e12 |
| `encoder.layer0.self_attn.out_proj` | 2.74e10 | — |

**放大集中在第一层与输入投影，深层相对很小** —— 说明梯度主要沿浅层路径回传，且在每个 `self_attn` 内部被数值放大。

### 2.4 反向逐层放大（决定性证据）

对同一 batch 注册 `full_backward_hook`：

| 模块位置 | epoch 0 grad_input | epoch 3 grad_input | epoch 3 单层放大约数 |
|---|---:|---:|---:|
| layer7.self_attn | 5.74e-3 | ≈1e-5 | — |
| layer5.self_attn | 3.85e-3 | 6.34e0 | ~1e5 |
| layer4.self_attn | 1.36e-2 | 2.78e2 | ~1.5e5 |
| layer3.self_attn | 6.85e-3 | 1.78e4 | ~1.9e5 |
| layer2.self_attn | 1.96e-2 | 4.83e6 | ~1.1e6 |
| layer1.self_attn | 5.41e-2 | 1.78e9 | ~2e6 |
| layer0.self_attn | 6.86e-1 | **1.75e13** | ~4.9e5 |

epoch 0 的逐层放大是温和的（0.006 → 0.69，约 120 倍）；epoch 3 变成每层 ~1e5–1e6，8 层累计 ~1e18。

### 2.5 “4.48 → 417 → 9.06e9 → 1.10e14”发生在哪里

- **增长点不是某一个参数，而是每个 `self_attn` 模块内部**：MultiheadAttention 的 `bmm` 反向与 softmax 反向在**巨大 V 值 + 饱和 softmax**的数值条件下产生放大。
- **第一个异常在一轮训练内就出现**：epoch 0 结束时第 0 层 QK score std = 3,893、`attn max = 1.000`（完全 one-hot），但此时梯度仍只有 4.48。
- **epoch 1 全程处于 LR 峰值平台（2.96–3.00e-4）**，梯度升到 417；随后正向反馈把 `in_proj_weight` 从 90.7 推到 258（epoch 2）、291（epoch 3），把 `norm1.bias` 从 2.27 推到 14.36，进一步加深饱和。
- **epoch 2/3 的 1e9/1e14 是同一放大机制的指数延续**。epoch 1 的权重没有保存（只保留 best/last），无法对 epoch 1 逐模块回放；但 epoch 0 → epoch 3 的逐层放大因子（590 → 4.9e5）与总梯度的增长（4.48 → 1.1e14）在量级上一致。

### 2.6 权重范数增长（参数更新后）

| 参数 | epoch 0 | epoch 2 | epoch 3 | 倍数 |
|---|---:|---:|---:|---:|
| 全部权重总范数 | 366.09 | 673.44 | — | 1.84×（ep2） |
| `encoder.layer0.self_attn.in_proj_weight` | 90.66 | 258.22 | 291.12 | 3.2× |
| `encoder.layer0.norm1.bias` | 2.27 | 14.36 | 14.25 | 6.3× |
| `encoder.layer3.norm1.bias` | 0.61 | 11.59 | — | **19.1×** |
| `cls_token` | 2.76 | 11.76 | — | 4.3× |
| `encoder.layer0.linear1.weight` | 69.39 | 100.10 | — | 1.44× |
| `encoder.layer0.linear2.weight` | 57.18 | 92.40 | — | 1.62× |
| `head.weight` | 1.09 | 1.35 | — | 1.24× |

注意 `head` 几乎不变，而 Transformer 内部权重显著膨胀：**输出层没有崩，崩的是表示层与注意力**。这也解释了为什么 Dev 输出会坍缩成一个常数类别。

### 2.7 注意力与残差流实测

| 指标 | epoch 0 | epoch 3 |
|---|---:|---:|
| L0 `norm1` 输出 std | 1.018 | 1.378 |
| L0 Q / K / V std | 18.0 / 13.0 / 5.98 | **120.1 / 108.0 / 16.7** |
| L0 score std | 3.89e3 | **4.36e5** |
| L0 score absmax | 1.20e5 | **6.46e6** |
| L0 `attn max` | **1.000** | **1.000** |
| L0 attention 输出 std | 59.2 | 245.8 |
| 残差流 std L0 → L7 | 153 → **504** | 1,217 → **4,231** |
| 残差流 absmax L7 | 6.2e3 | 1.6e4 |

`norm1` 每次输出 std≈1（归一化正常），但 Linear 的输出 std 达到 120 —— 说明**权重本身已把极少数方向放大**，随后 QK 点积进入 1e6 量级，softmax 完全 one-hot。

### 2.8 LayerNorm 是否异常

实测 per-token std（LayerNorm 输入）在 epoch 0 为 **78–350**，epoch 3 更大，**不存在 std→0 导致 1/σ 放大**的情况。LayerNorm 的 backward 反而以 ~1/σ 缩小梯度（grad_input/grad_output ≈ 0.25）。**LayerNorm 不是放大源。**

---

## 3. 学习率

### 3.1 实现核对

```python
OneCycleLR(optimizer, max_lr=3e-4, total_steps=20*9804=196080,
           pct_start=0.03, anneal_strategy="cos", cycle_momentum=False)
```

- `scheduler.step()` 在每个 `optimizer.step()` 之后调用一次；`accum=1`，**无累积与调度步数不一致**。
- epoch 3 结束时 `global_step = 39,216 = 4 × 9,804`，scheduler 的 `last_epoch` 与之相同。
- 实现本身正确，**问题在超参选择**。

### 3.2 实际 LR 曲线（复现自同一 PyTorch 版本）

```
total_steps = 196,080, warmup_steps = 5,882 (3%)
step 0..9   : 1.200000e-05 ... 1.200166e-05   （余弦 warmup，前段几乎不变）
step 90..99 : 1.216637e-05 ... 1.220130e-05
step 5882   : 3.000000e-04                     （峰值）
```

| epoch | 起始 LR | 结束 LR | 区间最小 | 区间最大 |
|---|---:|---:|---:|---:|
| 0 | 1.200e-5 | 2.997e-4 | 1.200e-5 | **3.000e-4** |
| 1 | 2.997e-4 | 2.962e-4 | 2.962e-4 | 2.997e-4 |
| 2 | 2.962e-4 | 2.888e-4 | 2.888e-4 | 2.962e-4 |
| 3 | 2.888e-4 | 2.778e-4 | 2.778e-4 | 2.888e-4 |

**关键时间对齐**：epoch 0 的平均梯度 4.48 包含了 5,882 步的低 LR warmup；epoch 1 全程处于 2.96–3.00e-4 平台，梯度立即跃升到 417。**LR 进入峰值的时刻，就是异常开始的时刻。**

### 3.3 3e-4 是否适合 hidden=1408 / 8 层 / FFN=5376

**不适合。** 依据：

1. 同一份代码、同一数据、同一 LR，d=192/L=1 与 d=352/L=2 的历史模型 grad_norm 始终 < 2.4；D=1408/L=8 时 epoch 0 已是 4.48（2–10 倍），epoch 1 达 417。
2. 该模型是**从头训练**（无预训练、无 LayerScale、无 QK-norm、无 embedding LayerNorm），对 LR 的容忍度远低于标准 Pre-LN 训练配方。
3. 211M 参数、8 层残差流在 3e-4 下每步 AdamW 更新幅度 ≈ lr；一个 epoch 9,804 步即可累计移动 ~2.9（初始权重行范数仅 ~0.7），足以把 attention 权重推入饱和区。
4. 3% warmup（5,882 步）对 8 层模型偏短，且 PyTorch OneCycle 的 warmup 是余弦曲线，前 90 步 LR 仅从 1.2e-5 升到 1.22e-5，随后才加速——相当于**长期低 LR 后突然进入高 LR 平台**，对稳定性最不利。

### 3.4 三种 LR 的理论稳定性差异

| LR | 预期 | 依据 |
|---|---|---|
| **3e-4（当前）** | **不稳定**。QK logits 快速进入 1e3–1e6，softmax 饱和，残差流膨胀，梯度正反馈 | 实测 |
| **1e-4** | 边界可用。约为当前峰值 1/3，权重漂移速度降到 1/3；但仍需配合输入归一化和更长的 warmup。历史小模型在 3e-4 下 gn<2.4，按参数量/深度保守估计 1e-4 可把 TS 的 gn 控制在个位数 | 量级推算 + 历史对照 |
| **5e-5** | 最保守。训练更慢（可能需要 30–40 epoch），但能最大程度避免饱和；适合作为**稳定性验证**的首轮配置 | 量级推算 |

推荐：**首轮用 5e-5–7.5e-5 做 1–2 epoch 稳定性验证，确认 grad_norm 稳定在个位数后再决定是否放大到 1e-4**。

---

## 4. 类别预测坍缩

### 4.1 真实类别分布

| 集合 | n | Stop | Forward | TurnLeft | TurnRight |
|---|---:|---:|---:|---:|---:|
| 全部 train pool | 1,279,882 | 18,521 (1.45%) | 946,031 (**73.92%**) | 157,164 (12.28%) | 158,166 (12.36%) |
| Train（剔除 2% Dev episode） | 1,254,793 | 18,151 (1.45%) | 927,278 (**73.90%**) | 154,223 (12.29%) | 155,141 (12.36%) |
| Dev（370 episodes） | 25,089 | 370 (1.47%) | 18,753 (**74.75%**) | 2,941 (11.72%) | 3,025 (12.06%) |

**确认：Forward 占 train 73.90% / Dev 74.75%**（用户提到的 74.7% 是 Dev 的数字）。

### 4.2 各 epoch 的 Dev 预测分布与指标

| epoch | 预测分布 S/F/L/R | acc | macro-F1 | weighted-F1 |
|---|---|---:|---:|---:|
| 0 | 272 / 24301 / 486 / 30 | 0.7562 | 0.3989 | 0.6713 |
| 1 | 89 / 24324 / 675 / 1 | 0.7493 | 0.3171 | 0.6643 |
| 2 | 0 / 25076 / 1 / 12 | 0.7470 | 0.2138 | 0.6393 |
| 3 | **0 / 25089 / 0 / 0** | 0.7475 | 0.2139 | 0.6394 |

### 4.3 混淆矩阵（行 = 真实，列 = 预测；S/F/L/R）

**epoch 0**
```
[[  179,  189,    2,   0],
 [   64, 18490,  191,  8],
 [   11,  2648,  282,  0],
 [   18,  2974,   11, 22]]
```

**epoch 1**
```
[[  51,  319,    0,  0],
 [  30, 18400,  323,  0],
 [   5,  2590,  346,  0],
 [   3,  3015,    6,  1]]
```

**epoch 2**
```
[[   0,  370,    0,  0],
 [   0, 18742,   1, 10],
 [   0,  2939,    0,  2],
 [   0,  3025,    0,  0]]
```

**epoch 3**
```
[[   0,  370,    0,  0],
 [   0, 18753,   0,  0],
 [   0,  2941,    0,  0],
 [   0,  3025,    0,  0]]
```

### 4.4 Per-class P / R / F1

| epoch | 类别 | P | R | F1 | support | predicted |
|---|---|---:|---:|---:|---:|---:|
| 0 | Stop | 0.6581 | 0.4838 | **0.5576** | 370 | 272 |
| 0 | Forward | 0.7609 | 0.9860 | 0.8589 | 18753 | 24301 |
| 0 | TurnLeft | 0.5802 | 0.0959 | 0.1646 | 2941 | 486 |
| 0 | TurnRight | 0.7333 | 0.0073 | 0.0144 | 3025 | 30 |
| 1 | Stop | 0.5730 | 0.1378 | 0.2222 | 370 | 89 |
| 1 | Forward | 0.7565 | 0.9812 | 0.8543 | 18753 | 24324 |
| 1 | TurnLeft | 0.5126 | 0.1176 | 0.1914 | 2941 | 675 |
| 1 | TurnRight | 1.0000 | 0.0003 | 0.0007 | 3025 | 1 |
| 2 | Stop | 0 | 0 | 0 | 370 | 0 |
| 2 | Forward | 0.7474 | 0.9994 | 0.8552 | 18753 | 25076 |
| 2 | TurnLeft | 0 | 0 | 0 | 2941 | 1 |
| 2 | TurnRight | 0 | 0 | 0 | 3025 | 12 |
| 3 | Forward | 0.7474 | 1.0000 | 0.8555 | 18753 | 25089 |
| 3 | 其余三类 | 0 | 0 | 0 | — | 0 |

**结论：epoch 3 已完全退化为多数类预测器。** Dev accuracy 0.7475 恰好等于 Forward 的先验比例（18,753 / 25,089 = 74.75%），模型在 Dev 上没有任何有效判别力。

### 4.5 Class weight / Label smoothing / Weighted sampling / 标签映射

| 项目 | 配置 | 影响 |
|---|---|---|
| class weight | **无**（`cross_entropy` 默认） | Stop 仅 1.45%，坍缩到 Forward 的 loss 代价很低（Forward 全预测时 loss ≈ 0.98，而均匀分布 loss = ln4 = 1.386） |
| label smoothing | **0.0** | 不提供额外正则 |
| weighted sampling | **无**（episode-block shuffle，均匀） | 少数类每 epoch 只出现约 1.45% |
| 标签映射 | `CLASSES = [Stop, Forward, TurnLeft, TurnRight]`，`labels.npy` 0–3 与 metrics 一致 | **正确**，未见错位 |

这不是实现错误，但**放大了坍缩的后果**：饱和/梯度爆炸后，模型收敛到“全 Forward”的局部极小，loss 仍然有限，早停机制无法把它拉回。

---

## 5. 104-token 输入检查

### 5.1 Token 组成与拼接顺序（`models.py::TransformerPreset.forward`）

| 段 | token 数 | 内容 | mask |
|---|---:|---|---|
| CLS | 1 | `cls_token + type_emb[0]` | 永不 padding |
| Current | 49 | `visual_proj(49×2048) + current_pos + type_emb[1]` | 永不 padding |
| Goal | 1 | `goal_proj(target_emb) → goal_current_attn(query=goal, kv=cur)`，经 `gc_norm + gc_ffn + gc_ffn_norm` | 永不 padding |
| History | 16 | `hist_vis_proj + hist_pose_proj + hist_act_proj + hist_time + type_emb[3]` | `history_mask < 0.5` |
| State | 1 | `state_proj(belief[7] + match[5]) + type_emb[4]` | 永不 padding |
| Landmark | 8 | `lm_proj(lm_feats) + type_emb[5]` | `lm_mask`（True=pad） |
| Relation | 28 | `rel_proj(rel_feats) + type_emb[6]` | `rel_mask`（True=pad） |
| **合计** | **104** | | |

`pad` 的拼接顺序与 `seq` 完全一致：`51 (cls+cur+goal) + 16 (hist) + 1 (state) + 8 (lm) + 28 (rel) = 104`。**对齐正确。**

### 5.2 逐项确认

| 检查项 | 结论 |
|---|---|
| 总数 104 | ✓（1+49+1+16+1+8+28） |
| 历史 16 步时间顺序 | `dataset_v2.py::history_block`：`for k in 1..16: j = t-k`，**索引 0 = t−1（最近），索引 15 = t−16（最早）**；严格因果（只用 j<t） |
| `hist_time` embedding | `nn.Parameter(torch.zeros(16, d))`，与上述索引一一对应；零初始化 |
| type embedding | `nn.Embedding(7, d)`，索引 0–6 = cls/vis/target/hist/state/lm/rel，与拼接顺序一致 ✓ |
| Padding mask | `src_key_padding_mask`，**True = 屏蔽**；`hist_pad = history_mask < 0.5`、`lm_mask = ~(w>0)`、`rel_mask = ~(valid>0.5)` 语义一致 ✓ |
| 是否错误屏蔽有效 token | 否：cls/current/goal/state 恒为 False；landmark/relation 的无效槽在 `landmark_geometry` 中已显式置 0 并 mask ✓ |
| 各模态数值范围 | 见 5.3 |
| 各模态 LayerNorm / 归一化 | **仅 landmark 的 `dist/50`、`bearing/π` 有归一化；其余输入无归一化** |
| Position embedding | `current_pos (49,d)` + `hist_time (16,d)`；cls/goal/state/lm/rel 无位置编码（它们各自只有 1 个或固定的相对顺序） |

### 5.3 真实 batch 的输入统计（128 个 state，epoch 3 数据顺序）

| 模态 | shape | mean | std | min | max | NaN/Inf |
|---|---|---:|---:|---:|---:|---:|
| `current` | (128,49,2048) | 0.108 | 0.699 | 0 | **41.31** | 0 |
| `target_emb` | (128,512) | 0.026 | 0.036 | 0 | 0.240 | 0 |
| `history_vis` | (128,16,2048) | 0.006 | 0.019 | 0 | 0.512 | 0 |
| `history_pose` | (128,16,3) | **−8.25** | **17.19** | **−72.99** | 30.00 | 0 |
| `history_action` | (128,16,4) | 1.008 | 0.725 | 0 | 3 | 0 |
| `history_mask` | (128,16) | 0.801 | 0.399 | 0 | 1 | 0 |
| `belief`（10 维，取 7 维） | (128,10) | **29.61** | **60.81** | −88.93 | **243.06** | 0 |
| `match`（5 维） | (128,5) | −3.45 | 15.37 | −48.49 | 45.84 | 0 |
| `states` | (128,3) | 418.0 | 374.4 | −2.91 | **1198.88** | 0 |
| `landmarks` | (128,8,3) | 377.6 | 336.6 | 0.004 | **1048.0** | 0 |

派生 token 的初始范数（epoch 0，实测）：

| token | norm | 备注 |
|---|---:|---|
| cls | 426 | `type_emb[0]` 主导 |
| current | 619 | `visual_proj` std 2.2 + `type_emb[1]` |
| goal | 444 | |
| history | 619 | `hist_pose_proj` std 7.2 + `type_emb[3]` |
| **state** | **13,170** | `state_proj` std 26.1，`type_emb[4]` 被大输入主导 |
| landmark | 503 | |
| relation | 431 | |

**根因证据：`state` token 的初始范数是 `cls` 的 31 倍，`history` 的 21 倍。** 直接原因是 `state_proj` 的输入含未归一化的 `belief`（dist/bearing 达 243、`unc=50`、`since=99`）与 `match`（达 48）；`hist_pose` 含 −73～30 的原始量。

虽然 `norm1` 是 per-token LayerNorm，会把每个 token 归一化到 std≈1，但**输入尺度失衡仍然影响：**
1. 初始时 `state` token 在残差流中占绝对主导，注意力容易集中到它；
2. `state_proj` / `hist_pose_proj` 的梯度在 epoch 0 就已是最大（46.4 / 17.0）；
3. 模型为放大弱信号（visual/landmark）不得不增大投影权重，进入“权重增大 → QK 增大 → softmax 饱和”的正反馈。

**未修改数据集**（按要求）。

---

## 6. Transformer 结构检查

`models.py::TransformerPreset(1408, 8, 8, 3*1408=4224, 5376)`：

| 项目 | 实现 | 评价 |
|---|---|---|
| Pre-LN / Post-LN | **Pre-LN**（`norm_first=True`） | 正确选择 |
| Residual | 标准 Pre-LN：`x = x + attn(norm1(x))`；`x = x + ffn(norm2(x))` | 结构正确，但**残差流无范数约束** |
| Q/K/V/O 投影 | `nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)`；单独 `out_proj` | 默认 xavier 初始化 |
| FFN 激活 | GELU | 正常 |
| LayerNorm eps | 默认 `1e-5` | 正常 |
| Dropout | 0.1（encoder + gc_ffn + goal_current_attn + head 前无 dropout） | 偏小，对 211M 模型不足以抑制饱和 |
| Embedding 初始化 | `type_emb = nn.Embedding(7, 1408)` → **N(0,1)**；`cls_token` / `current_pos` / `hist_time` = **零** | `type_emb` 每个向量范数 ≈ 37，**主导 token 范数**；旧 `TransformerS4` 用的是零初始化 `current_type/goal_type` |
| Linear 初始化 | PyTorch 默认 `xavier_uniform_` | 对 8 层 Pre-LN 偏大；无 LayerScale/DeepNorm |
| 输出头初始化 | `nn.Linear(d, 4)` 默认 | 正常；实测 head 权重几乎不变 |
| Embedding LayerNorm | **无** | 缺失稳定化 |
| QK-norm | **无** | 缺失稳定化 |
| LayerScale | **无** | 缺失稳定化 |

**结论：结构本身没有实现错误，但把 d=192/L=1 时代的“裸 Pre-LN”直接放大到 d=1408/L=8，同时沿用旧的初始化与 LR，缺少现代大模型常用的稳定化措施。**

需要的调整（本轮只分析，不修改）：

1. **输入/嵌入归一化**：`state`/`hist_pose`/`belief`/`match` 做固定尺度标准化（例如按数据集统计的 mean/std，或除以经验尺度 100/50），或在投影后加 LayerNorm；
2. **Pre-LN 保留，但增加 LayerScale**（每层残差分支乘可学习 γ，初始 1e-4~1e-2），或使用 DeepNorm；
3. **QK-norm**（对 Q/K 做 LayerNorm），直接抑制 `score` 到 1e6 的量级；
4. **更小的初始 LR + 更长 warmup**（建议峰值 5e-5~1e-4，warmup 8–10%）；
5. **梯度裁剪保留**，但增加“异常步跳过”（`gn > 1e4` 时 skip，而不是仅处理非有限值）；
6. **训练监控**：每步记录 `grad_norm` 到 jsonl，设置告警阈值（例如 >100 持续 100 步）；
7. **类别不平衡**：可选 `class_weight ∝ 1/freq`（Stop 权重约 17×），或 focal loss；至少增加 weighted-F1 监控。

---

## 7. 恢复方案比较

| 方案 | 保留进度 | 是否重训 | 是否重新初始化 | 预计时间 | 对导航能力的影响 |
|---|---|---|---|---|---|
| **A. 从 last.pt（epoch 3）恢复 + 降 LR** | 保留 3 个 epoch（但权重已损坏） | 否 | 否 | ~12–14 h | **不可行**：epoch 3 权重在 Dev 上 100% 预测 Forward，`in_proj_weight` 291、`norm1.bias` 14.3、梯度 1e14，已进入坏区域；继续训练只会沿同一正反馈 |
| **A′. 从 best.pt（epoch 0）恢复 + 降 LR** | 仅 1 个 epoch | 是（从 epoch 1 起） | **需要重建 optimizer/scheduler**（best.pt 不含优化器状态） | ~13 h | 可行但不理想：epoch 0 已出现 `attn max=1.0`、score std 3893 的饱和迹象；低 LR 可能把模型拉回，但残差流已增长到 504 |
| **B. 从头重训 + 稳定化配置** | 无（放弃 4 个 epoch） | 是 | 是 | ~13–16 h（可先跑 1–2 epoch 验证） | **推荐**。加入输入归一化、LR 5e-5~1e-4、warmup 8–10%、（可选）LayerScale/QK-norm、异常步跳过；可同时缓解类别坍缩 |
| **C. 仅修“发现的代码/数据 bug”** | 无 | 是 | 是 | — | **不适用**：标签映射、mask、loss、token 数、时间顺序均未发现实现错误；真正的问题是尺度与超参不适配，属于 B 的范畴 |

**推荐：B（从头重训）**，并先做一次 **1–2 epoch 的稳定性冒烟测试**：
- 通过标准：`grad_norm < 10` 且不随步数持续上升；`attn max < 0.9`；残差流 std 不随层数单调爆炸；
- 冒烟通过后再放长到 20 epoch。

**不推荐 A**：epoch 3 的权重不可用；epoch 0 的权重虽有 0.3989 的 Dev F1，但已出现饱和前兆，且缺少 optimizer state，恢复成本与重训接近。

---

## 8. 是否需要重新训练

**需要。**

- epoch 3 的 `last.pt`：Dev 全 Forward，macro-F1 = 0.2139，**不可用于部署或闭环评测**；
- `best.pt`（epoch 0）：Dev macro-F1 = 0.3989、weighted-F1 = 0.6713，可作为**临时基线**，但 TurnRight F1 仅 0.0144、TurnLeft 0.1646，且已出现注意力饱和前兆，不建议直接用于最终 Val-Seen 评测；
- 当前 Stage 14 的闭环评测**尚未执行**（Supervisor 已在 `TRAIN FAILED` 处停止），因此没有产生错误的评测结果。

---

## 9. 是否需要检查其他四种模型的共同代码

**共同代码需要加固，但不需要因此重跑 Small/Medium。**

### 9.1 已训练完成的模型

| 模型 | 状态 | 结论 |
|---|---|---|
| Small | 17 epoch，Dev F1 0.5532–0.5771，grad_norm 0.385–10.76（稳定） | **正常，无需重跑** |
| Medium | 17 epoch，Dev F1 0.4990–0.6025，grad_norm 1.90–18.99（稳定） | **正常，无需重跑** |

### 9.2 尚未训练的模型

| 模型 | 配置 | 风险 |
|---|---|---|
| **Transformer-M** | `TransformerM12Fix(d=2048, L=10, H=16, ffn=7424)`，约 498M 参数 | **中高风险**。其骨架（v5fix, d=256/L=4）历史 grad_norm 0.57–1.43，稳定；但 Stage 14 的配置大了约 100 倍，且沿用相同的 LR=3e-4 与未归一化输入。**必须先做 1 epoch 冒烟测试**，监控 `grad_norm < 10` |
| **Large** | `TransformerPreset(2560, 12, 16, 10240, 10240)`，约 1.03B 参数 | **最高风险**。与 TS **完全同架构**，只是更大更深，**几乎必然复现同样的崩溃**。必须在 TS 的稳定化方案验证通过后再训练，且 LR 应更低（≤5e-5） |

### 9.3 共同代码的具体建议

`train_expert.py` 的训练循环本身没有实现错误，但有两处防护不足，**影响全部五个模型**：

1. **异常梯度只判 finite，不判量级**：建议增加
   ```python
   if not math.isfinite(gn) or gn > 1e4:   # 跳过病态更新并计数
       nan_updates += 1
       optimizer.zero_grad(set_to_none=True)
       continue
   ```
2. **只在每 1000 步打印 loss/acc，不记录 grad_norm**：建议每个 update 追加
   `{"step":..., "grad_norm_preclip": gn, "lr":...}` 到 `grad_trace.jsonl`，并在 `status.json` 中加入 `grad_norm_ema`，便于 watchdog 告警。

> 注意：以上是**建议**，本轮未修改任何训练代码。Shared 数据管线（`dataset11b.py`、`build_blank_mask`、`apply_blank`）与 `v5fix_contract` 的语义检查均未发现错误。

---

## 10. 复现命令

本报告中的实测数据由以下只读脚本产生（均未修改训练代码、未写 checkpoint）：

```bash
# 权重范数对比（epoch0 best vs epoch2 last）
python diag_grad_ts.py  checkpoints/Transformer-S/best.pt /tmp/grad_epoch0.json 3 4
python diag_grad_ts.py  checkpoints/Transformer-S/last.pt /tmp/grad_epoch3.json 3 4
# Dev 评估（混淆矩阵 / per-class / macro / weighted F1）
python diag_dev_ts.py   checkpoints/Transformer-S/best.pt /tmp/dev_epoch0.json
python diag_dev_ts.py   checkpoints/Transformer-S/last.pt /tmp/dev_epoch3.json
# 注意力内部量
python diag_attn.py     checkpoints/Transformer-S/best.pt
python diag_attn.py     checkpoints/Transformer-S/last.pt
# LayerNorm 输入 per-token 方差
python diag_ln.py       checkpoints/Transformer-S/best.pt
# 反向逐层梯度流
python diag_bwdflow.py  checkpoints/Transformer-S/best.pt
python diag_bwdflow.py  checkpoints/Transformer-S/last.pt
```

（脚本位于 `/tmp/ts_diag/`，如需长期保留可移入 `diagnosis/scripts/`。）

---

## 11. 最终答复（对应提问清单）

1. **崩溃原因**：输入尺度失衡 + 3e-4 峰值 LR 过大 → 注意力饱和（`attn max=1.0`，score 达 1e6）→ 残差流爆炸（std 154→4231）→ 反向逐层放大约 1e3–1e6 → 梯度 1e14 → clip+AdamW 继续推动权重 → 正反馈 → 坍缩到多数类。
2. **梯度异常**：统计口径为**裁剪前**；裁剪执行正确；`accum=1` 无除错；loss 无重复加权；无重复反向；放大发生在每个 `self_attn` 内部；LayerNorm 不是放大源；权重范数显著增长（`in_proj` 3.2×、`norm1.bias` 6–19×）。
3. **学习率**：OneCycle 实现正确；epoch 0 为 1.2e-5→2.997e-4（含 5,882 步余弦 warmup），epoch 1 全程 2.96–3.00e-4；scheduler/optimizer 步数一致（epoch 3 末 39,216）。3e-4 对该规模**偏大**；1e-4 边界可用；5e-5 最稳。
4. **类别坍缩**：Forward 占 train 73.90% / Dev 74.75%；epoch 3 预测 0/25089/0/0，acc 恰等于多数类先验；无 class weight / label smoothing / weighted sampling；标签映射正确。
5. **104-token**：总数、拼接顺序、mask 对齐、时间顺序（索引 0 = t−1）、type embedding、mask 语义均正确；**问题是输入未归一化**（state token 范数是 cls 的 31 倍）。
6. **Transformer 结构**：Pre-LN、GELU、eps=1e-5、dropout=0.1、默认初始化；**缺少** embedding LN / LayerScale / QK-norm；`type_emb` 的 N(0,1) 主导 token 范数。
7. **最推荐恢复方案**：**B —— 从头重训 + 稳定化配置**（LR 5e-5~1e-4、warmup 8–10%、输入归一化、异常步跳过、监控 grad_norm；可选 LayerScale/QK-norm），先跑 1–2 epoch 冒烟验证。
8. **是否需要重新训练**：**需要**（epoch 3 不可用；epoch 0 仅可作临时基线）。
9. **其他模型**：Small/Medium 正常无需重跑；**TM 必须先冒烟**；**Large 风险最高，必须先采用稳定化方案**；共同训练循环建议增加 grad_norm 监控与异常步跳过。

---

**诊断结束。未启动任何新训练，等待你的确认后再执行恢复方案。**
