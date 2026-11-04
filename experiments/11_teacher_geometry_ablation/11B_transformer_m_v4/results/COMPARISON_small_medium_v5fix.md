# Small / Medium / V5fix 三模型统一对比（Val-Seen，修正协议）

生成时间: 2026-11-04 10:40 CST（seed42）
协议: 11A High+Medium 教师池 18,521 eps / 1,279,882 states；batch 128；全量 epoch；
≤30 epoch；patience 10；Val-Seen Macro-F1 选 best；闭环 2,498 任务、STOP 半径 20m、
最大 200 步；**无候选任务计入 SR/OSR 分母**。

## 训练与 Val 指标

| 模型 | 架构 | 参数 | 训练耗时 | best Epoch | Train Loss@best | Train Acc@best | Val Macro-F1 | Val Acc |
|---|---|---|---|---|---|---|---|---|
| Small | MeanPoolMLP（四动作） | 2.24M | 4.4 h（17 ep 早停） | 6 | 0.579 | 77.8% | **0.7243** | 79.8% |
| Medium | GoalCrossAttentionMEC（四动作） | 2.24M | 5.8 h（12 ep 早停） | 1 | 0.625 | 76.8% | **0.7103** | 79.1% |
| V5fix | TransformerM12Fix（修正几何+空白合成） | 2.58M | 6.3 h（14 ep 早停） | 3 | 0.527 | 78.9% | **0.7475** | 81.0% |

## Val-Seen 闭环（2,498 任务）

| 模型 | SR | OSR | NE (m) | SPL | 平均步数 | 失败分类（stop_reason） |
|---|---|---|---|---|---|---|
| Small | **34.67%** (866) | 35.19% | 91.6 | 0.327 | 63.0 | out_of_bounds 1203 / model_stop 1214 / step_cap 81 |
| Medium | **23.98%** (599) | 25.62% | 105.1 | 0.238 | 54.9 | out_of_bounds 1707 / model_stop 789 / step_cap 1 / collision 1 |
| V5fix | **53.60%** (1339) | 54.08% | 56.2 | 0.469 | 71.1 | out_of_bounds 372 / model_stop 1895 / step_cap 231 |
| （旧 v4 基线） | 46.76% | — | — | — | — | model_stop 1373 / out_of_bounds 964 |
| （旧 v5 基线） | 60.09% | 61.8% | 57.3 | 0.496 | — | model_stop 1807 / out_of_bounds 311 |

## 要点

1. **V5fix（修正协议下）闭环 SR 53.6%**：六项修正（几何 token mask/相对角/归一化/
   per-pair 屏蔽/空白合成/单次编码）自旧 v4 的 46.8% 显著提升，接近旧 v5 基线
   60.1% 且失败构成大幅改善（越界 964→372）。
2. **Small/Medium 明显弱于 Transformer 系列**：闭环 SR 34.7%/24.0%，主要失败为
   越界（1203/1707）——非 Transformer 架构在长程闭环定位上劣势明显。
3. 三模型 Val Macro-F1 接近（0.710-0.748），但闭环 SR 差距大，说明 Macro-F1
   对闭环性能的预测力有限，最终以闭环指标为准。

## 权重核验

各模型 best.pt SHA256 见同目录 `SHA256_MANIFEST.md`。
