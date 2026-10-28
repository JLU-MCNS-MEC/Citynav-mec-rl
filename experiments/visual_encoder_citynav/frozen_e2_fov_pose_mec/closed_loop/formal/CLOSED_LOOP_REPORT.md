# 实验 A 闭环导航评估（冻结 E2-FOV-Pose + MEC）

- 冻结编码器 SHA-256：`441de539f1d30f2be68d77a1c954daef5f3ab3bd3501f33b2c9798f89517011b`
- 决策：MEC 六类 logits 的 argmax（与 token 验证一致）
- 成功：终止位置距目标 ≤ 20 m；oracle 成功：rollout 中最小距离 ≤ 20 m
- 每划分 24 条、最多 100 步、8 分片；样本量小，置信区间宽，不作显著性声明
- Test-Unseen 未读取

| split | model | variant | n | SR | oracle SR | SPL | NE (m) | nDTW | steps | stop rate | s/step |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val_seen | small | full | 24 | 0.000 | 0.042 | 0.000 | 225.5 | 0.550 | 100.0 | 0.000 | 5.23 |
| val_seen | small | map_free | 24 | 0.000 | 0.000 | 0.000 | 230.2 | 0.407 | 17.0 | 1.000 | 4.56 |
| val_seen | transformer_m | full | 24 | 0.000 | 0.042 | 0.000 | 219.0 | 0.580 | 100.0 | 0.000 | 5.09 |
| val_seen | transformer_m | map_free | 24 | 0.000 | 0.042 | 0.000 | 222.9 | 0.277 | 15.1 | 1.000 | 6.90 |
| val_unseen | small | full | 24 | 0.000 | 0.083 | 0.000 | 195.8 | 0.577 | 100.0 | 0.000 | 6.91 |
| val_unseen | small | map_free | 24 | 0.000 | 0.042 | 0.000 | 202.5 | 0.476 | 18.0 | 1.000 | 4.98 |
| val_unseen | transformer_m | full | 24 | 0.000 | 0.083 | 0.000 | 189.5 | 0.595 | 100.0 | 0.000 | 6.83 |
| val_unseen | transformer_m | map_free | 24 | 0.000 | 0.083 | 0.000 | 194.8 | 0.283 | 17.7 | 1.000 | 7.13 |

## 解读要点

- `full` 变体在 token 验证集上 Stop 召回为 0，闭环中同样不停止，因而全部跑满步数；SR 与 oracle SR 的差异说明轨迹是否曾接近目标。
- `map_free` 变体能够 Stop，但转向类动作识别差，路径质量需结合 SPL/nDTW 看。
- 渲染成本约 3–8 s/步（取决于地图密度与并行度），是本次评估规模的主要限制。
