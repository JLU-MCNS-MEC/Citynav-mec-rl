# 五种 MEC 专家模型计算量反推与最小架构修改方案

生成日期：2026-11-17　分支：`exp/citynav-mec-fixed50m-topdown`
范围：**只读审计 + FLOPs 核算 + 架构参数反推 + 理论可行性分析**。
未训练任何模型，未跑 Val-Seen 闭环，未修改模型定义或 Checkpoint，未修改
Stage 12 正式结果。

---

## 0. 结论摘要（先看这一段）

1. 现有五种模型的真实前向计算量分别为
   **Small 4.473 MFLOPs、Medium 68.225 MFLOPs、Transformer-S 119.123 MFLOPs、
   Transformer-M 774.068 MFLOPs、Large 650.998 MFLOPs**。
2. 与 5 / 15 / 40 / 100 / 200 GFLOPs 目标相差 **1,118× / 220× / 336× /
   129× / 307×**。
3. **不存在“只改一个参数、保持架构与权重、达到 ±5%”的方案**。唯一能在
   ±5% 内命中的单参数方案是 hidden dimension 扩大 **12.2–48.3×**；FFN 宽度
   需要扩大 **228–28,480×**，层数需要扩大 **143–557×**。这些都不是“最小修改”，
   而是新架构。
4. 在两个参数均受合理上限（F≤4×、L≤4×、H≤4×、D≤2×）约束时，最好的候选
   只能达到目标的 **0.43%–8.10%**，无法接近 ±5%。
5. 命中目标所需的 hidden 维度方案会使参数量达到 **414M–3,111M**，其中
   **99.8%–99.9% 的参数需要重新初始化**；原 Checkpoint 只能继承极少部分。
6. 因此本报告的正式建议是：**不要按 5/15/40/100/200 GFLOPs 强行放大现有五种
   架构**。要么把 Stage 12 的时延建模计算量改为实测值（或按实测值重新标定
   档位），要么接受“新架构 + 全量重训”，这已超出“最少参数修改”的范围。

---

## 1. 计算口径

- batch size = 1；每次导航决策一次完整前向。
- FLOPs = 2 × MACs。
- 统计：所有 `nn.Linear`、`nn.MultiheadAttention`（q/k/v/out 投影、
  attention score、weighted sum）、FFN、Embedding（lookup 无 MACs）。
- 未统计：LayerNorm、GELU、Softmax、Dropout、残差加法等逐元素运算
  （在 GFLOPs 量级下可忽略；本报告在方法上明确排除）。
- 不含 UAV 视觉编码器、数据预处理、无线通信。
- 实现：`scripts/analyze_model_flops.py`，对每个 `nn.Linear` /
  `nn.MultiheadAttention` 注册 forward hook；为让
  `nn.TransformerEncoderLayer` 的模块 hook 生效，临时替换为其 slow Python
  forward（`_sa_block` + `_ff_block`），避免 PyTorch fused fast path 绕过统计。
- 交叉核对：Small 4.473M 与 Stage 11 的 4.473M 完全一致。Medium 68.22M /
  TM 774.07M 高于 Stage 11 的 55.12M / 555.96M，差额正是旧 hook 未统计的
  MHA 内部 q/k/v 投影与 attention score；本报告按“包含 Attention”的完整口径
  统计。

## 2. 现有模型真实 FLOPs 与参数

| 模型 | 参数量 | 真实 FLOPs | GFLOPs | 目标 GFLOPs | 目标/当前 |
|---|---:|---:|---:|---:|---:|
| Small | 2,239,236 | 4,473,344 | 0.00447 | 5 | **1,117.7×** |
| Medium | 2,635,268 | 68,224,512 | 0.06822 | 15 | **219.9×** |
| Transformer-S | 1,644,292 | 119,122,560 | 0.11912 | 40 | **335.8×** |
| Transformer-M | 4,630,276 | 774,068,224 | 0.77407 | 100 | **129.2×** |
| Large | 6,128,676 | 650,997,952 | 0.65100 | 200 | **307.2×** |

模块占比（完整明细见 `results/module_flops_breakdown.csv`）：

| 模型 | 主导模块 |
|---|---|
| Small | head 47.0%、current 投影 23.4%、history 视觉投影 23.4% |
| Medium | current 投影 75.3%、goal→current cross-attention 19.3%、head 2.7% |
| Transformer-S | fusion encoder **60.1%**（self-attn 27.1% + FFN 33.0%）、visual_proj 32.3%、cross-attention 6.2% |
| Transformer-M | fusion encoder **90.0%**（4 层 self-attn 34.0% + FFN 56.0%）、vis_proj 6.6%、hist_vis 2.2% |
| Large | fusion encoder **85.0%**（2 层 self-attn 30.6% + FFN 54.4%）、visual_proj 10.8%、cross-attention 3.8% |

关键事实：**Transformer 类模型的 FLOPs 主要由 fusion encoder 的
self-attention + FFN 决定；Small/Medium 主要由视觉投影与 head 决定。**
因此扩大 FFN 宽度或层数对 Transformer 类最直接，但它需要的倍数（228–1,014×）
远超合理范围。

## 3. 输入契约审计（104-token 问题）

| 模型 | 当前 token 结构 | 是否 104-token |
|---|---|---|
| Small | 8 路 late-fusion MLP；current/target/history/state/lm/rel 各自 mean-pool 或投影，无 token 序列 | **否** |
| Medium | 1 个 goal query 对 49 个 current token 做 cross-attention；history 1 个 pooled 向量；无 encoder | **否**（无 104-token 序列） |
| Transformer-S | cls 1 + current 49 + goal 1 + history **1（16 步 mean-pool）** + state 1 + lm 8 + rel 28 = **89** | **否** |
| Transformer-M | cls 1 + current 49 + target 1 + history **16（独立 token）** + state 1 + lm 8 + rel 28 = **104** | **是** |
| Large | 同 Transformer-S = **89** | **否** |

共同点（五种模型完全一致）：49 个 current 视觉 token（49×2048）、512 维目标
描述子、12 维去重状态、8 个 landmark、28 个 pairwise relation、V5fix 修正后的
几何特征/mask/归一化（`TransformerM12Fix.landmark_geometry` 或等价实现）。

统一到 104 tokens 需要：

- **Transformer-S / Large**：把 history 从“16 步 masked mean-pool 的 1 个
  token”改为“16 个独立 history token”，增加 `hist_time`（16×h）时间嵌入，
  前缀 mask 从 53 改为 68，fusion encoder 输入从 89 改为 104。代码改动集中在
  `model11_explicit.py` 的 `forward` 与新增参数；视觉/目标/状态/几何/关系
  不变。注意：这改变了 history 的聚合语义（mean-pool → per-step），不仅是
  “输入维度”变化。
- **Medium**：当前没有 token 序列，要满足 104-token 需新增一个 Transformer
  encoder（新层、新参数、重新初始化），属于架构改动，不是最小修改。
- **Small**：当前是 MLP，要满足 104-token 需改为 token 编码 + encoder，属于
  架构改动。

**不要为了增加 FLOPs 而改变输入模态语义。** 104-token 统一本身不会把任何模型
带到 5–200 GFLOPs 量级（见第 4 节）。

## 4. 架构参数反推（A/B/C）

定义：

- A1：只改 FFN 宽度（Transformer 类）或 fusion MLP 宽度（Small/Medium）。
- A2：只改 encoder 层数（仅 Transformer 类）。
- A3：只改 hidden dimension（最后手段）。
- B：两个参数，均在合理上限内（F≤4×、L≤4×、H≤4×、D≤2×）。
- C：达到目标所需的多参数（这里以 hidden 维度为主）。

### 4.1 Small（当前 D=256, H=512）

| 方案 | 修改 | 新值 | 预测 FLOPs | 偏差 | 参数量 | 倍数 | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| A1 | fusion MLP H | 1,217,746 | 5.00 G | 0% | 2,501 M | 2,378× | 不可接受 |
| A2 | encoder layers | — | — | — | — | — | Small 无 encoder |
| A3 | hidden D | 12,356 | 5.00 G | 0% | 3,111 M | 48.3× | 命中但非最小 |
| B | D×2, H×4 | 512 / 2,048 | 21.54 M | −99.57% | 10.78 M | — | 远低于目标 |
| C | D=12,356 | 12,356 | 5.00 G | 0% | 3,111 M | 48.3× | 新架构，99.98% 新参数 |

### 4.2 Medium（当前 D=256, F_gc=512, H=512）

| 方案 | 修改 | 新值 | 预测 FLOPs | 偏差 | 参数量 | 倍数 | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| A1 | gc_ffn 宽度 | 14,581,812 | 15.00 G | 0% | 7,482 M | 28,480× | 不可接受 |
| A2 | encoder layers | — | — | — | — | — | Medium 无 encoder |
| A3 | hidden D | 7,548 | 15.46 G | +3.04% | 1,288 M | 29.5× | 命中但非最小 |
| B | D×2, H/Fgc×4 | 512 / 2,048 | 178.92 M | −98.81% | 12.88 M | — | 远低于目标 |
| C | D=7,548 | 7,548 | 15.46 G | +3.04% | 1,288 M | 29.5× | 新架构，99.95% 新参数 |

### 4.3 Transformer-S（当前 D=192, L=1, F=576）

| 方案 | 修改 | 新值 | 预测 FLOPs | 偏差 | 参数量 | 倍数 | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| A1 | FFN 宽度 | 584,039 | 40.00 G | 0% | 226 M | 1,014× | 不可接受 |
| A2 | encoder layers | 557.2 | 40.00 G | 0% | 208 M | 557× | 不可接受 |
| A3 | hidden D | 4,435 | 40.24 G | +0.59% | 414 M | 23.1× | 命中但非最小 |
| B | F×4, L×4 | 2,304 / 4 | 806.68 M | −97.98% | 5.41 M | — | 远低于目标 |
| C | D=4,435 | 4,435 | 40.24 G | +0.59% | 414 M | 23.1× | 新架构，99.90% 新参数 |

### 4.4 Transformer-M（V5FIX，当前 D=256, L=4, F=1,024）

| 方案 | 修改 | 新值 | 预测 FLOPs | 偏差 | 参数量 | 倍数 | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| A1 | FFN 宽度 | 233,957 | 100.00 G | 0% | 483 M | 228× | 不可接受 |
| A2 | encoder layers | 572.1 | 100.00 G | 0% | 453 M | 143× | 不可接受 |
| A3 | hidden D | 3,126 | 100.00 G | 0% | 523 M | 12.2× | 命中但非最小 |
| B | F×4, L×4 | 4,096 / 16 | 8.10 G | −91.90% | 39.3 M | — | 仅 8.1% 目标 |
| C | D=3,126 | 3,126 | 100.00 G | 0% | 523 M | 12.2× | 新架构，99.78% 新参数 |

### 4.5 Large（当前 D=352, L=2, F=1,408）

| 方案 | 修改 | 新值 | 预测 FLOPs | 偏差 | 参数量 | 倍数 | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| A1 | FFN 宽度 | 796,819 | 200.00 G | 0% | 1,128 M | 566× | 不可接受 |
| A2 | encoder layers | 724.8 | 200.00 G | 0% | 1,084 M | 362× | 不可接受 |
| A3 | hidden D | 6,657 | 200.71 G | +0.35% | 1,627 M | 18.9× | 命中但非最小 |
| B | F×4, L×4 | 5,632 / 8 | 6.54 G | −96.73% | 38.9 M | — | 仅 3.3% 目标 |
| C | D=6,657 | 6,657 | 200.71 G | +0.35% | 1,627 M | 18.9× | 新架构，99.91% 新参数 |

完整机器可读结果见 `results/candidate_architectures.csv`。

## 5. 原 Checkpoint 权重继承分析

| 方案 | 可完整继承 | 可部分继承 | 需新增初始化 | 动作头 |
|---|---:|---:|---:|---|
| A1（FFN/head 加宽） | 非 FFN/head 参数；占新模型 **0.03%–0.72%** | 旧 FFN/head 权重子块；**0.01%–0.24%** | **99.04%–99.96%** | D 不变时不变；Small A1 的 head 第二层需扩展（部分） |
| A2（增加层） | 旧参数（非新层）；占新模型 **0.56%–1.02%** | 0 | 新层 **98.98%–99.44%** | 不变（D 不变） |
| A3（hidden 扩大） | 无（所有 D 维张量形状改变） | 重叠子矩阵 **0.02%–0.22%** | **99.78%–99.99%** | 部分重初始化（新输入列；输出 4 不变） |
| B（F×4, L×4 等） | **8.8%–22.7%** | **2.9%–7.6%** | **69.8%–88.2%** | D 不变的 Transformer 不变；Small/Medium D×2 时部分 |
| C（目标命中） | 0 | 0.02%–0.22% | 99.78%–99.99% | 部分重初始化 |

要点：

- A1 的“完整继承”比例看似存在，但相对于命中目标后的巨大新模型，只有
  0.1% 左右；新增参数几乎全部是随机初始化。
- A2 的所有旧参数都能继承，但新模型有 99% 的参数来自新增层。
- A3/C 没有形状完全一致的张量；只有重叠子矩阵可复制，动作头输入列需要
  重新初始化（输出维度仍为 4，不需要整体重训动作头，但需要重训融合部分）。
- **不能声称“可以无损继承”**。完整的、逐张量的 shape 兼容性审计见
  `results/weight_transfer_audit.csv`（近似比例，按张量重叠计算）。

## 6. 八个问题的回答

1. **真实 FLOPs**：4.473M / 68.225M / 119.123M / 774.068M / 650.998M
   （Small/Medium/TS/TM/Large）。
2. **与目标差距**：1,118× / 220× / 336× / 129× / 307×。
3. **只改一个参数能否接近目标？** 不能以“合理修改”做到。唯一在 ±5% 内的
   单参数方案是 hidden D 扩大 12.2–48.3×（参数量 414M–3,111M）；FFN/层数
   单参数需要 143–28,480×，更不合理。
4. **两个参数？** 在合理上限（F≤4×、L≤4×、H≤4×、D≤2×）下没有一个模型能
   进入 ±5%；最好只达到目标的 0.43%–8.10%。
5. **更多参数？** 命中目标需要 hidden D 3,126–12,356（12.2–48.3×）或
   557–725 层 / 234k–797k FFN 宽度；参数量 414M–3,111M，99.8%–99.9% 新初始化。
6. **对原始闭环影响最小的方案？** 保持当前五个架构与 Checkpoint 不变（零
   影响）；如果必须改，B 方案（F×4、L×4）影响相对最小，但仍需重训且只能
   提升到目标的 0.4%–8.1%，无法满足 5–200 GFLOPs。
7. **Checkpoint 能继承多少？** 见第 5 节：A1/A2/A3 的完整继承比例
   0.07%–1.02%，新增初始化 99% 以上；B 方案完整继承 8.8%–22.7%，新增
   69.8%–88.2%；C 方案几乎全部新初始化。
8. **是否存在不适合当前架构的目标？** 是。全部五个目标都比当前架构大
   2–3 个数量级；Small/Medium 是 MLP，TS/Large 是 89-token、TM 是 104-token
   的小型 Transformer，均无法通过“少量参数修改”达到 GFLOPs 级计算量。
   此外当前 TM（774M）> Large（651M），与目标 TM 100G < Large 200G 的排序
   方向相反；若要求档位单调，Large 需要比 TM 更大的扩张。

## 7. 建议

**建议 A（推荐，零架构改动）**：保留五种已训练模型与 Checkpoint，把 Stage 12
时延建模计算量改为**实测 FLOPs**（0.00447 / 0.0682 / 0.119 / 0.774 / 0.651
GFLOPs），或按实测值重新标定“轻/中/重”档位。这样不改变任何导航行为，也不
需要重训；MEC 时延差异将由节点算力与通信主导（与当前 Stage 12 的结论一致）。

**建议 B（若 GFLOPs 档位是硬约束）**：接受“新架构 + 全量重训”，而不是
“最小参数修改”。按第 4 节 A3/C 的 hidden 维度或 A1/A2 的极端 FFN/层数设计
新模型；必须明确 99% 以上参数是新初始化的，原 Checkpoint 只能作为初始化
的一部分，不能声称无损继承。

**建议 C（折中，需你确认）**：先用 B 方案（F×4、L×4）把计算量提高约
2–8×（TM 8.10G、Large 6.54G、TS 0.81G、Medium 0.18G、Small 0.022G），
再重新标定目标档位；这仍然需要重训，且 70%–88% 参数是新初始化的，但比
A1/A3 的极端方案更接近“可训练”的范围。

**不建议**把 A1/A3 的具体数值（1.2M 宽 MLP、12k hidden、557–725 层、
234k–797k FFN）当作实际实现方案；它们只是数学上命中目标所需的参数，
不代表可训练的架构。

## 8. 产物与复现

```
experiments/13_mec_architecture_sizing/
├── MEC_ARCHITECTURE_FLOPS_DESIGN.md
├── scripts/
│   ├── analyze_model_flops.py          # hook-based exact FLOPs
│   ├── search_candidate_architectures.py
│   └── make_figures.py
├── results/
│   ├── baseline_flops.json
│   ├── baseline_model_flops.csv
│   ├── module_flops_breakdown.csv
│   ├── candidate_architectures.csv
│   └── weight_transfer_audit.csv
└── figures/
    ├── flops_target_vs_current.png
    ├── params_comparison.png
    └── module_flops_breakdown.png
```

复现：

```bash
PY=/home/tenant2/miniconda3/envs/citynav-mec/bin/python
$PY experiments/13_mec_architecture_sizing/scripts/analyze_model_flops.py
$PY experiments/13_mec_architecture_sizing/scripts/search_candidate_architectures.py
$PY experiments/13_mec_architecture_sizing/scripts/make_figures.py
```

本阶段未修改任何模型定义或 Checkpoint；未训练；未跑 Val-Seen 闭环；未修改
Stage 12 结果。Stage 13 文件尚未提交/推送，等待确认。
