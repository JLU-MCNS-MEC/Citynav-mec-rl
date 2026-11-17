# Stage 13-R1：五种 MEC 模型统一输入契约修正与架构参数重新反推

生成日期：2026-11-17　分支：`exp/citynav-mec-fixed50m-topdown`
基线：Stage 13 原始审计 `d2280be6`（保留不动）
本报告目录：`experiments/13_mec_architecture_sizing/revision_r1/`

范围：只读审计、模型实例化、随机输入单次前向、FLOPs profiling；**未修改任何
模型定义或 Checkpoint，未训练，未跑 Val-Seen 闭环，未改动 Stage 12 结果。**

---

## 0. 结论摘要

1. 统一输入契约后，三个 Transformer 的输入均为 **104 tokens**：
   CLS 1 + Current 49 + Target 1 + History 16 + State 1 + Landmark 8 +
   Relation 28。Transformer-S / Large 由 89 → 104（history 不再 mean-pool）。
2. 104-token 后 FLOPs：
   - Transformer-S：**144.242 MFLOPs**（原 119.123，+21.09%）；
   - Transformer-M：**774.068 MFLOPs**（已满足 104，不变）；
   - Large：**770.062 MFLOPs**（原 650.998，+18.29%）。
   Small / Medium 已接收全部模态（允许内部 mean-pool / cross-attention），
   FLOPs 不变：4.473 M / 68.225 M。
3. 与 5 / 15 / 40 / 100 / 200 GFLOPs 目标仍相差 **1,118× / 220× / 277× /
   129× / 260×**。
4. **仍然不存在“合理参数范围内达到 ±5%”的方案**。合理上限
   （F≤4×、L≤4×、H≤4×、D≤2×）内的两参数方案只能达到目标的
   **0.43%–8.10%**。单参数方案中所需倍数最小的是 hidden 扩大
   12.2–48.3×（参数量 414M–3,111M，新增初始化 99% 以上）；FFN 宽度需
   228–28,480×，层数需 143–616 层——所有单参数 ±5% 方案都不是最小修改。
5. **本轮最推荐的最小修改**是：Transformer-S / Large 采用 16 个独立
   history tokens（89→104），新增 `hist_time` 与统一的 `type_emb`；该修改
   **不改变 hidden/层数/FFN**，encoder 与 history 投影权重完全兼容，
   原 Checkpoint 约 **99.7%–99.9% 可完整继承**。它满足输入契约，但只把
   TS/Large 的 FLOPs 提高 21.09% / 18.29%，仍远低于 40/200 GFLOPs。

---

## 1. 表 1：输入契约统一前后对比

| 模型 | 原输入处理 | 修正后输入处理 | 是否 Transformer | 是否满足统一信息契约 |
|---|---|---|---|---|
| Small | current mean-pool；target/state/lm/rel 投影；history vis/pose/action masked mean-pool；late-fusion MLP | 不变（内部 mean-pool 允许） | 否 | **是**（8 种模态齐全） |
| Medium | 1 goal query + 49 current KV cross-attention；history mean-pool；state/lm/rel 投影；late fusion | 不变（内部 cross-attention 允许） | 否 | **是**（8 种模态齐全） |
| Transformer-S | CLS 1 + current 49 + goal 1 + history **1（pooled）** + state 1 + lm 8 + rel 28 = **89** | CLS 1 + current 49 + target 1 + history **16** + state 1 + lm 8 + rel 28 = **104** | 是 | **修正后满足** |
| Transformer-M | 104 tokens（已满足） | 不变 | 是 | **是（基准）** |
| Large | 同 Transformer-S = **89** | 同 Transformer-S = **104** | 是 | **修正后满足** |

共同点（五种模型完全一致）：49×2048 current、512 维 target、12 维去重状态、
8×3 landmark、28×3 relation、V5fix 几何（`TransformerM12Fix.landmark_geometry`）、
`history_mask < 0.5`（True=pad）、同一四动作定义。

**注意**：统一 104 tokens 不等于要求 Small/Medium 变成 Transformer；它们只需
接收相同信息。Small/Medium 没有 encoder，不虚构 layers/heads。

## 2. 表 2：原始模型与 104-token 统一后计算量

| 模型 | 原始 FLOPs | 统一输入后 FLOPs | 增加比例 | 目标 | 目标/当前（统一后） |
|---|---:|---:|---:|---:|---:|
| Small | 4.473 M | 4.473 M | 0.00% | 5 G | 1,117.7× |
| Medium | 68.225 M | 68.225 M | 0.00% | 15 G | 219.9× |
| Transformer-S | 119.123 M | **144.242 M** | **+21.09%** | 40 G | **277.3×** |
| Transformer-M | 774.068 M | 774.068 M | 0.00% | 100 G | 129.2× |
| Large | 650.998 M | **770.062 M** | **+18.29%** | 200 G | **259.7×** |

TS/Large 的增加来自两部分：

1. **History 投影从 1 次变 16 次**：`hist_vis_proj`（2048→h）、
   `hist_pose_proj`（3→h）、`hist_act_proj`（4→h）分别对 16 个历史步执行；
   增加量 = 15 × (2048+3+4) × h × 2 FLOPs。
2. **Encoder 输入 89→104**：self-attention 的 Q/K/V/O、score、weighted sum
   与 FFN 的 token 维从 89 变 104。

参数变化：TS +3,840（16h `hist_time` + 7h `type_emb` − 3h 旧
`current_type`/`goal_type`/`goal_pos`，h=192）；Large +7,040（h=352）。
这些 embedding 只改变参数，不增加 matmul FLOPs。

## 3. 表 3：架构参数反推（A/B/C）

说明：A1/A2/A3 的“预测 FLOPs”按目标反解，数值上命中目标，但所需参数值
不可接受；B 在合理上限内，但远离目标。Small/Medium 的 layers/heads 标记为
“不适用”，不虚构 Transformer 参数。

| 模型 | 方案 | hidden | layers | heads | FFN / MLP | 目标 | 预测 FLOPs | 偏差 |
|---|---|---|---:|---:|---|---:|---:|---:|
| Small | A1 fusion width | 256 | — | — | 1,217,746 | 5 G | 5.00 G | 0% |
| Small | A2 | — | — | — | — | — | — | 无 encoder |
| Small | A3 hidden | 12,356 | — | — | 24,712 | 5 G | 5.00 G | 0% |
| Small | B (D×2,H×4) | 512 | — | — | 2,048 | 5 G | 21.54 M | −99.57% |
| Small | C | 12,356 | — | — | 24,712 | 5 G | 5.00 G | 0% |
| Medium | A1 gc FFN | 256 | — | — | 14,581,812 | 15 G | 15.00 G | 0% |
| Medium | A2 | — | — | — | — | — | — | 无 encoder |
| Medium | A3 hidden | 7,430 | — | — | 14,860 | 15 G | 15.00 G | ~0% |
| Medium | B (D×2,H/Fgc×4) | 512 | — | — | 2,048 | 15 G | 178.92 M | −98.81% |
| Medium | C | 7,430 | — | — | 14,860 | 15 G | 15.00 G | ~0% |
| Transformer-S | A1 FFN | 192 | 1 | 4 | 499,571 | 40 G | 40.00 G | 0% |
| Transformer-S | A2 layers | 192 | 470 | 4 | 576 | 40 G | 40.00 G | 0% |
| Transformer-S | A3 hidden | 4,121 | 1 | 4 | 12,363 | 40 G | 40.20 G | +0.51% |
| Transformer-S | B (F×4,L×4) | 192 | 4 | 4 | 2,304 | 40 G | 951.27 M | −97.62% |
| Transformer-S | C | 4,121 | 1 | 4 | 12,363 | 40 G | 40.20 G | +0.51% |
| Transformer-M | A1 FFN | 256 | 4 | 8 | 233,957 | 100 G | 100.00 G | 0% |
| Transformer-M | A2 layers | 256 | 572.1 | 8 | 1,024 | 100 G | 100.00 G | 0% |
| Transformer-M | A3 hidden | 3,126 | 4 | 8 | 12,506 | 100 G | 100.00 G | 0% |
| Transformer-M | B (F×4,L×4) | 256 | 16 | 8 | 4,096 | 100 G | 8.10 G | −91.90% |
| Transformer-M | C | 3,126 | 4 | 8 | 12,506 | 100 G | 100.00 G | 0% |
| Large | A1 FFN | 352 | 2 | 8 | 681,689 | 200 G | 200.00 G | 0% |
| Large | A2 layers | 352 | 616 | 8 | 1,408 | 200 G | 200.00 G | 0% |
| Large | A3 hidden | 6,172 | 2 | 8 | 24,690 | 200 G | 200.61 G | +0.30% |
| Large | B (F×4,L×4) | 352 | 8 | 8 | 5,632 | 200 G | 7.665 G | −96.17% |
| Large | C | 6,172 | 2 | 8 | 24,690 | 200 G | 200.61 G | +0.30% |

关键读数：

- **A1/A2** 的所需倍数：Small head 2,378×；Medium gc_ffn 28,480×；
  TS FFN 867× / 470 层；TM FFN 228× / 143 层；Large FFN 484× / 308 层。
- **A3** 的 hidden 倍数：Small 48.3×、Medium 29.0×、TS 21.5×、TM 12.2×、
  Large 17.5×；参数量 414M–3,111M。
- **B** 是唯一“架构合理”的候选，但只能达到目标的 0.43%–8.10%，**不存在
  同时合理且 ±5% 的方案**。

## 4. 表 4：Checkpoint 继承

### 4.1 仅做 104-token 输入统一（不改 FFN/层数/hidden）

| 模型 | 原参数量 | 候选参数量 | 完整继承 | 部分继承 | 新初始化 | 说明 |
|---|---:|---:|---:|---:|---:|---|
| Small | 2,239,236 | 2,239,236 | 100% | 0% | 0% | 输入模态已齐全，不变 |
| Medium | 2,635,268 | 2,635,268 | 100% | 0% | 0% | 输入模态已齐全，不变 |
| Transformer-S | 1,644,292 | 1,648,132 | **1,643,716（99.73%）** | 384（0.02%） | 4,032（0.24%） | encoder/history 投影形状不变；新增 hist_time/type_emb；旧 goal_pos 丢弃 |
| Transformer-M | 4,630,276 | 4,630,276 | 100% | 0% | 0% | 已是 104-token 基准 |
| Large | 6,128,676 | 6,135,716 | **6,127,620（99.87%）** | 704（0.01%） | 7,392（0.12%） | 同上 |

- Encoder 权重：**形状完全兼容**（token 数不在参数维度中），可完整继承。
- History 三个投影：形状完全兼容（2048→h、3→h、4→h），只是从“对 pooled
  1 个向量执行”改为“对 16 个向量逐 step 执行”，**权重可完整复用**。
- 新增 `hist_time`（16h）与统一 `type_emb`（7h）；旧 `current_type`/
  `goal_type`（2h）可复制到 `type_emb` 的 current/target 行（部分继承）；
  旧 `goal_pos`（1h）在新规则中没有对应位置（不可继承/丢弃）。
- 动作头：D 不变，**不需要重新初始化**。

### 4.2 进一步扩大 FFN / 层数 / hidden

| 方案 | 完整继承 | 部分继承 | 新初始化 | 动作头 |
|---|---:|---:|---:|---|
| A1（FFN/head 加宽） | 0.03%–0.72% | 0.01%–0.24% | 99.1%–99.96% | D 不变时不变；Small A1 head 第二层部分 |
| A2（增加层） | 0.67%–1.02% | 0 | 99.0%–99.3% | 不变 |
| A3/C（hidden 扩大） | 0 | 0.05%–0.22% | 99.85%–99.95% | 部分重初始化（新输入列；输出 4 不变） |
| B（合理两参数） | 8.8%–22.8% | 2.9%–7.6% | 69.5%–88.2% | Transformer D 不变时不变；Small/Medium D×2 时部分 |

## 5. 表 5：最终推荐架构（最小修改）

| 模型 | 推荐方案 | hidden | layers | heads | FFN / MLP | 预测 FLOPs | 相对原模型 | 继承 |
|---|---|---|---:|---:|---|---:|---:|---:|
| Small | **保持原架构**（已满足信息契约） | 256 | — | — | 512 | 4.473 M | 0% | 100% |
| Medium | **保持原架构**（已满足信息契约） | 256 | — | — | 512 | 68.225 M | 0% | 100% |
| Transformer-S | **16 history tokens（89→104）** | 192 | 1 | 4 | 576 | **144.242 M** | +21.09% | **99.73%** |
| Transformer-M | **保持 V5FIX 104-token** | 256 | 4 | 8 | 1,024 | 774.068 M | 0% | 100% |
| Large | **16 history tokens（89→104）** | 352 | 2 | 8 | 1,408 | **770.062 M** | +18.29% | **99.87%** |

推荐理由：

1. **输入契约统一的最小改动是 history tokens**：它满足 104-token 契约，
   不改变 hidden/层数/FFN，encoder 与 history 投影权重完全兼容；新增参数
   仅 16h+7h（TS 0.24%、Large 0.12%），动作头不变。
2. **Small/Medium 不需要改**：它们已经接收全部 8 种模态，内部 mean-pool /
   cross-attention 是允许的；强行加 encoder 反而会破坏原有闭环能力与权重继承。
3. **GFLOPs 目标仍不可达**：104-token 统一只把 TS/Large 提高 21.09%/18.29%，
   与 40/200 GFLOPs 仍差 277×/260×。若目标是 5–200 GFLOPs，必须接受
   A3/C 级别的 hidden 扩张（12–48×，99%+ 新参数）或重新标定目标。
4. **如果只需“合理扩算力”**：B 方案（F×4、L×4）把计算量提高约 2–8×
   （TS 951M、TM 8.10G、Large 7.665G、Medium 178.9M、Small 21.54M），
   但需要重训，且 70%–88% 参数新初始化；仍远低于 5–200 GFLOPs。

## 6. 对五个问题（你要求展示的）的直接回答

1. **修正后的输入契约**：见第 1 节表 1。三个 Transformer 统一 104 tokens；
   Small/Medium 接收相同 8 种模态，内部处理方式不同但信息不缺失。
2. **104-token 后三个 Transformer FLOPs**：TS **144.242 MFLOPs**、
   TM **774.068 MFLOPs**、Large **770.062 MFLOPs**。
3. **参数反推方案**：见第 3 节表 3（A/B/C）。没有“合理且 ±5%”的方案；
   B 方案只能到目标的 0.43%–8.10%。
4. **原 Checkpoint 继承比例**：
   - 仅做 104-token 统一：TS **99.73% 完整继承**、Large **99.87% 完整继承**；
     Small/Medium/TM **100% 不变**。
   - 进一步扩大架构：A1 完整继承 0.03%–0.72%、A2 0.67%–1.02%、
     A3/C 0%、B 8.8%–22.8%；新增初始化 69.5%–99.96%。
5. **最推荐的最小修改**：TS/Large 采用 16 个独立 history tokens（89→104），
   新增 `hist_time` 与统一 `type_emb`；Small/Medium/TM 保持原架构。该方案
   满足 104-token 契约，权重继承约 99.7%–99.9%，但只把 TS/Large 的 FLOPs
   提高 21.09%/18.29%，**不能达到 40/200 GFLOPs**。若 GFLOPs 档位是硬约束，
   需要另选 A3/C 或重新标定目标，并接受 99%+ 新初始化。

## 7. 产物

```
experiments/13_mec_architecture_sizing/revision_r1/
├── MEC_ARCHITECTURE_FLOPS_DESIGN_R1.md
├── results/
│   ├── input_contract_audit.csv
│   ├── revised_baseline_flops.csv
│   ├── revised_baseline_flops.json
│   ├── revised_candidate_architectures.csv
│   ├── revised_weight_transfer_audit.csv
│   └── module_flops_breakdown.csv
├── figures/
│   ├── token_contract_comparison.png
│   ├── revised_flops_comparison.png
│   └── candidate_architecture_comparison.png
└── scripts/
    ├── analyze_revised_architectures.py
    ├── search_revised_candidates.py
    └── make_revision_figures.py
```

复现：

```bash
PY=/home/tenant2/miniconda3/envs/citynav-mec/bin/python
$PY experiments/13_mec_architecture_sizing/revision_r1/scripts/analyze_revised_architectures.py
$PY experiments/13_mec_architecture_sizing/revision_r1/scripts/search_revised_candidates.py
$PY experiments/13_mec_architecture_sizing/revision_r1/scripts/make_revision_figures.py
```

本阶段未修改任何模型定义或 Checkpoint；未训练；未跑 Val-Seen 闭环；未修改
Stage 12 结果；未覆盖 Stage 13 原报告。等待你确认架构参数后再开始修改与重训。
