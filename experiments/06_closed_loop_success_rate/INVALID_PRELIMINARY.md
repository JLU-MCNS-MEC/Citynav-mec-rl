# 作废说明：2026-10-29 的初步 Stage 06 结果

状态：**INVALID / 不得引用**

`results/closed_loop_val_seen_seed42.jsonl`、`results/summary_val_seen_seed42.json`、
`results/selection_val_seen.json` 来自 2026-10-29 22:55–22:56 的 3 条 episode 冒烟测试，
对应的 checkpoint 来自 Stage 05 的作废初步训练（见
`../05_transformer_m_training/INVALID_PRELIMINARY.md`）。

作废原因：

1. checkpoint 由旧候选库与旧教师生成的特征训练得到；
2. 任务集合只有 3 条，且长度分组边界来自旧教师步数分布；
3. 实验 03B 尚未产出，短/中/长分组定义不成立。

记录（不是结论）：3 条 episode，SR 0，STOP 时平均目标距离 110.9 m，
`Forward` 126 次、`TurnLeft` 10 次、`TurnRight` 9 次、`Stop` 3 次。

正式的闭环验证必须在实验 03B 与修正后的实验 04、05 完成后重跑。
