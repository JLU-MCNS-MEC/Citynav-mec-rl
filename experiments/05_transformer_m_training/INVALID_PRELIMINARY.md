# 作废说明：2026-10-29 的初步 Stage 05 产物

状态：**INVALID / 不得引用、不得复用**

本目录下的 `features/`、`features_scaled/`、`runs/`、`checkpoints/` 与 `results/*.log`
是 2026-10-29 22:31–23:06（Stage 04 提交 `722b62b` 之后）的一次未完成尝试，产生于
以下前置条件尚未满足时：

1. 实验 04 的候选库使用未修复的高程有效性掩码（`height > -1000`，没有 nodata
   比例过滤），无效区域可以成为全局目标候选；
2. 训练样本来自实验 03 的人工轨迹教师（高质量人工轨迹子集），不是实验 03B 的
   直接目标导向教师；
3. 每个 episode 的 `belief`、`cand_idx`、`cand_sim`、`cand_rel` 都绑定旧候选库，
   候选库重建后全部失效；
4. Stage 04 报告声称当前视觉是 7×7×2048 空间特征，但 `encode_episode()` 实际返回
   2048 维全局池化；初版的验证数值不能作为输入契约的依据。

后果：特征缓存与 checkpoint 全部作废。原始文件**保留在服务器**用于对照，不删除、
不提交；重跑会使用新的 `features_v2/`、`candidates_v2/` 与 `runs_v2/` 目录。

初步结果仅作记录（不是结论）：两个种子 40 epoch 后 train acc = 1.000、
val acc ≈ 0.70–0.75、macro-F1 ≈ 0.46、Stop 召回 0.10–0.27；
3 条闭环 episode SR = 0。

恢复记录见 `../00_archive_and_token_cleanup/RECOVERY_2026-10-30.md`。
