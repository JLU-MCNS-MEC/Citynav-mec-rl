# Stage 15 · 阶段 B：共同状态下的专家行为互补性

- 共同状态数：**1000**（按教师动作分层采样，Stop 150 / Forward 450 / TurnLeft 200 / TurnRight 200）
- 五个冻结专家，同一输入，batch=1，顺序加载（无显存冲突）
- Small/Medium 使用原始预处理；TS/TM/Large 使用 Stage 14-R2 归一化

## B1. 单模型动作准确率与置信度

| 模型 | 教师动作准确率 | 平均置信度 | 平均熵 | 高置信错误 (>0.8) |
|---|---:|---:|---:|---:|
| Small | 83.7% | 0.845 | 0.388 | 37 (3.7%) |
| Medium | 68.1% | 0.789 | 0.533 | 84 (8.4%) |
| TS | 72.0% | 0.799 | 0.480 | 74 (7.4%) |
| TM | 69.2% | 0.782 | 0.528 | 80 (8.0%) |
| Large | 68.2% | 0.776 | 0.547 | 61 (6.1%) |

## B2. 专家间动作一致性

- 五专家完全一致：**697 / 1000 = 69.7%**
- 存在分歧：**303 / 1000 = 30.3%**

| 组合 | 一致率 | 分歧数 |
|---|---:|---:|
| Small vs Large | 75.4% | 246 |
| Small vs TM | 76.8% | 232 |
| Small vs Medium | 78.5% | 215 |
| Small vs TS | 79.2% | 208 |
| Medium vs Large | 89.9% | 101 |
| Medium vs TM | 90.0% | 100 |
| Medium vs TS | 91.0% | 90 |
| TS vs Large | 93.1% | 69 |
| TS vs TM | 93.8% | 62 |
| TM vs Large | 94.4% | 56 |

## B3. 按教师动作的分歧

- **Stop**（150 states）：分歧率 23.3%
- **Forward**（450 states）：分歧率 8.2%
- **TurnLeft**（200 states）：分歧率 57.5%
- **TurnRight**（200 states）：分歧率 58.0%

## B4. Stop 决策

| 模型 | Stop 状态上预测 Stop | 非 Stop 状态上误报 Stop |
|---|---:|---:|
| Small | 94.7% | 0.6% |
| Medium | 90.7% | 0.2% |
| TS | 93.3% | 0.2% |
| TM | 80.7% | 0.1% |
| Large | 78.7% | 0.1% |

## B5. 分歧状态中谁是正确的

| 正确专家数 | 状态数 |
|---|---:|
| 0 | 20 |
| 1 | 143 |
| 2 | 32 |
| 3 | 42 |
| 4 | 66 |

## 产物

- `shared_states_manifest.csv`
- `expert_action_predictions.csv`
- `expert_action_agreement.csv`
- `expert_disagreement_by_state.csv`
- `expert_action_accuracy.csv`
- `stop_decision_analysis.csv`
- `disagreement_winners.csv`
- 图：`figures/phaseB_*.png`
