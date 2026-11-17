# Stage 14 正式训练前审计：Smoke Test + 输入契约

日期：2026-11-17　结论：**全部通过**

| 模型 | 参数 | Batch×累积 | Smoke 更新步 | 104-token | 八模态全部使用 | Smoke 峰值 | R3 估计 | 判定 |
|---|---:|---|---:|---|---|---:|---:|---|
| Small | 61,121,028 | 128×1 | 100 | — | 是 | 1.64 GB | 2.05 GB | PASS |
| Medium | 166,011,908 | 128×1 | 100 | — | 是 | 4.04 GB | 4.39 GB | PASS |
| Transformer-S | 211,181,060 | 128×1 | 100 | 是 | 是 | 10.08 GB | 10.29 GB | PASS |
| Transformer-M | 498,567,684 | 128×1 | 100 | 是 | 是 | 19.19 GB | 19.19 GB | PASS |
| Large | 1,034,877,444 | 64×2 | 50 (100 micro) | 是 | 是 | 28.29 GB | 24.04 GB | PASS |

## 逐项检查

- **Small**：15/15 项通过
- **Medium**：15/15 项通过
- **Transformer-S**：15/15 项通过
- **Transformer-M**：15/15 项通过
- **Large**：15/15 项通过

## 输入 Shape 契约（八种模态）

| 模态 | Shape |
|---|---|
| Current Visual | 49×2048 |
| Target | 512 |
| History Visual | 16×2048 |
| History Pose | 16×3 |
| History Action | 16×4 (one-hot) |
| State | 12 (belief 7 + match 5) |
| Landmark | 8×3 |
| Relation | 28×3（由 V5FIX 几何层生成） |

三种 Transformer 的 token 序列为 1(cls)+49(current)+1(goal)+16(history)+1(state)+8(landmark)+28(relation)=**104**。

## 结论与注意

- Smoke 与正式训练共用 `models.build_model`、`collate11`/`to_model_batch11`，输入处理完全一致。
- Large 实测峰值 28.28 GB，高于 R3 估计 24.04 GB（+4.24 GB），仍低于 32 GB；正式训练沿用 batch 64 × 累积 2，不启用 gradient checkpointing，若 OOM 立即暂停报告。
- Dev 集：训练池 2% episode 独立留出（本阶段开始前已确认），与 Val-Seen 2498 闭环任务无重叠；不使用 Val-Seen 选 best。
