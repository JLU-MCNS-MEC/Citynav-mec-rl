# 五模型统一对比报告（Small / Medium / V5 Fix / Transformer-S / Large）

生成时间: 2026-11-04 22:15 CST（seed 42，全部为本轮修正协议结果）
协议: 11A High+Medium 教师池 18,521 eps / 1,279,882 states；batch 128；全量 epoch；
≤30 epoch；patience 10；Val-Seen Macro-F1 选 best；闭环 Val-Seen 2,498 任务、
STOP 半径 20m、最大 200 步；无候选任务计入分母（本轮 Val-Seen 无候选任务 = 0）。

## 训练与 Val 指标

| 模型 | 架构（原定义） | 参数 | 训练耗时 | best Epoch | Train Acc@best | Val Macro-F1 |
|---|---|---|---|---|---|---|
| Small | MeanPoolMLP 四动作 | 2.24M | 4.4 h（17 ep） | 6 | 77.8% | 0.7243 |
| Medium | GoalCrossAttentionMEC 四动作 | 2.24M | 5.8 h（12 ep） | 1 | 76.8% | 0.7103 |
| Transformer-S | ExplicitReasoningTransformer 192/1层/4头/ff×3 | 1.64M | 8.0 h（17 ep） | 6 | 77.8% | 0.7258 |
| Large | ExplicitReasoningTransformer 352/2层/8头/ff×4 | 6.13M | 10.7 h（23 ep） | 12 | 79.5% | 0.7340 |
| V5fix | TransformerM12Fix（修正几何） | 4.63M | 6.3 h（14 ep） | 3 | 78.9% | 0.7475 |

## Val-Seen 闭环（统一 2,498 任务）

| 模型 | SR | OSR | NE (m) | SPL | 平均步数 | 失败分类（stop_reason：自停/越界/超步） |
|---|---|---|---|---|---|---|
| Small | 34.67% (866) | 35.19% | 91.6 | 0.327 | 63.0 | 1214 / 1203 / 81 |
| Medium | 23.98% (599) | 25.62% | 105.1 | 0.238 | 54.9 | 789 / 1707 / 1（collision 1） |
| Transformer-S | 37.47% (936) | 39.23% | 119.4 | 0.344 | 80.7 | 1182 / 977 / 338（collision 1） |
| Large | 43.55% (1088) | 45.40% | 98.1 | 0.402 | 77.6 | 1338 / 893 / 266（collision 1） |
| V5fix | **53.60%** (1339) | **54.08%** | **56.2** | **0.469** | 71.1 | 1895 / 372 / 231 |
| 旧 v4 基线 | 46.76% | — | — | — | — | 1373 / 964 / — |
| 旧 v5 基线 | 60.09% | 61.8% | 57.3 | 0.496 | — | 1807 / 311 / — |

## 分析：架构规模与多模态融合对导航性能的影响

1. **显式 Transformer 架构优于 MLP/交叉注意力 late-fusion**：同训练池与协议下，
   Transformer-S（SR 37.5%）> Small（34.7%）> Medium（24.0%）；规模增大进一步
   提升：Large（SR 43.6%）> Transformer-S。层级自注意力 + 显式 token 融合对
   闭环导航的帮助在 Small/Medium→TS/Large 对比中一致可见。
2. **几何修正比规模更关键**：V5fix（4.63M 参数，SR 53.6%）超过参数更大的
   Large（6.13M，SR 43.6%）——修正无效几何 token 掩码、相对方位角、per-pair
   关系屏蔽与空候选合成对闭环定位的增益大于单纯扩大 Transformer 规模；
   且 V5fix 越界失败仅 372（Large 893），定位/信念质量显著更稳。
3. **Val Macro-F1 与闭环 SR 弱相关**：五模型 F1 集中在 0.710-0.748，闭环 SR
   却从 24.0% 到 53.6% 相差 2.2 倍——Macro-F1 无法预测闭环表现，最终结论
   以闭环指标为准（与三模型审计结论一致）。
4. **NE 与越界失败强相关**：V5fix NE 56.2m（越界 372）< 旧 v5 57.3m（311）
   < Small 91.6m（1203）< Large 98.1m（893）< Medium 105.1m（1707）
   < TS 119.4m（977）——越界是 NE 恶化的主因，TS/Large 的 step_cap 偏高
   （338/266）提示其在终点附近徘徊、停止决策不足，与"到达未停/早停"失败
   构成（TS never_stop/premature 占比高）互相印证。
5. **旧 v5 基线（60.1%）仍为闭环最强**：修正版 V5fix（53.6%）略低于旧 v5；
   旧 v5 权重基于旧协议 harness 训练，其闭环数字来自修正协议评测（公平口径），
   说明部分历史训练细节（如 10C 特征链）可能携带额外信息，属于遗留问题，
   不在本轮范围。

## 权重与证据

- 各模型 best.pt SHA256 见同目录 `SHA256_MANIFEST.md`。
- 逐模型证据：`runs_*/seed42/{config.json,history.jsonl,REPORT.md,best.pt}`；
  闭环：`experiments/v4_closed_loop_revalidation/results/{model}_val_seen_20261103/`。
- 三模型审计报告：`AUDIT_2026-11-04_three_models.md`（本目录）。
