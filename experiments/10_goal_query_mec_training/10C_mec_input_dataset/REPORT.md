# 10C 报告：MEC 输入模态确定与训练数据生成

- 日期：2026-10-31
- 服务器：新服务器 `ubuntu`（RTX 5090 32 GB）
- 分支：`exp/citynav-mec-fixed50m-topdown`
- 上承：10B 结论——最终视觉匹配方案 = **C1（09C-1）**
- 契约：`INPUT_CONTRACT.md`；泄露审计：`LEAK_AUDIT.md`

## 1. 最终 MEC 输入模态（10D/10E 使用）

| # | 组 | 形状 | 说明 |
|---|---|---|---|
| 1 | 当前视觉 | 49×2048 | 冻结 ResNet-50 layer4 空间特征（俯视局部 RGB，heading 帧） |
| 2 | 历史视觉 | 16×2048 | 因果历史窗口（t−16..t−1）池化嵌入 |
| 3 | 历史位姿/动作 | 16×3 + 16×4 + mask | 相对当前 heading 帧 |
| 4 | 目标查询描述子 | 512 | **C1 目标中心 2×2 块 layer2 描述子**（L2 归一） |
| 5 | 连续目标状态 | 10 | 09C ContinuousBelief：前向/侧向偏移、距离、方向、置信度、不确定性、后验熵、距上次更新步数、本步匹配 dx/dy |
| 6 | 当前匹配测量 | 5 | C1 本步输出：dx、dy、score、margin、sharpness |

**排除**：语言/BERT、网络状态、真实目标坐标直接输入、未来动作、成功标签、
评测器信息（详见 LEAK_AUDIT.md 逐项证据）。

## 2. 数据生成

| 项 | 内容 |
|---|---|
| 主分布 | Train-Seen 03B 教师轨迹 22,002 条（876,398 状态），因果循环生成特征 |
| 恢复分布 | C1 在全部 22,002 条 Train-Seen 任务上的闭环轨迹（4 进程 × ~1.5 ep/s，共 ~1 小时），标签 = C1 动作（类 DAgger，20% 配比） |
| 采样 | **逐状态均匀**（配额 ∝ 轨迹长度，随机相位步长，全局降采样到恰好 N）——修正 Stage 07 短轨迹过度代表缺陷（见 LEAK_AUDIT §5 验证） |
| 数据集 | 训练 250,000 状态（教师 200,000 + 恢复 50,000）；验证 1,564 状态（Val-Seen 400 条 × 4） |
| 因果一致性 | belief/match 沿轨迹用与 09C-1 相同的门控（score≥0.875 且两帧一致≤10 m）与 Kalman 参数（09C 校准冻结）演化 |
| 断点续跑 | episode 级 npz + SHA-256 journal 校验；采样与特征分离（采样重跑无需重算特征） |

## 3. 关键决策与论证

1. **目标特征用 C1 查询描述子（512-d 中心块）**，替代 Stage 05 的 2048-d
   全局目标嵌入：10B 证明 C1 的匹配表征是最终有效方案，MEC 应消费同一表征。
2. **belief 用 09C 连续状态（10-d）**，替代 Stage 05 的 12-d 候选 Top-5
   信念：含置信度/不确定性/熵/更新时效，是 C1 闭环控制实际使用的信息。
3. **恢复状态（DAgger 式）**：Stage 07 已确认训练/闭环协变量偏移（目标在
   身后 0.65% vs 20.6%）是旧 Transformer-M 失败主因；C1 恢复状态把模型
   暴露在真实闭环分布下。标签用 C1 动作（C1 是除教师外最强的现有策略，
   Train-Seen SR 由闭环测出），**无类别损失加权**（用户禁令）。
4. **cross-fitting 不适用**：C1 零可学习参数，无逐状态自拟合；belief 超参数
   是 Train-Seen 上校准的全局标量（等同学习率）。论证见 LEAK_AUDIT §3。
5. **layer2 匹配特征保持 batch-1**（与 09C 生产、10E 推理完全一致）；
   layer4 历史池化沿用 Stage 05 的 batch-128 缓存约定（10E 中历史为
   batch-1，数值级差异 ≤0.5，与 Stage 05/06 同款已知事项，已记录）。

## 4. 数据审计摘要

- 无坐标/未来/成功标签泄露（逐通道检查通过）
- 时间因果性：历史严格 t−16..t−1；belief 因果演化
- 采样分布与原始池一致（距离桶最大偏差 0.1pp）；短轨迹占比 15.7% = 池值 ✓
- 训练/推理特征提取代码同源（10C 构建器与 10E 闭环共用同一因果循环逻辑）

## 5. 产物

- `INPUT_CONTRACT.md`、`LEAK_AUDIT.md`
- `results/cache/{train_seen,val_seen,recovery}/`（episode npz + journal + index + states.csv + sample_stats.json；~35 GB，本地缓存不入库）
- `results/train_manifest.json`（22,002 条 Train-Seen 任务清单 + SHA）
- `results/c1_rollouts/`（C1 恢复轨迹 journal + episodes，含信念内部状态）
- scripts：`build_features_10c.py`、`build_recovery_10c.py`

## 6. 运行记录

- 教师特征：8 进程并行 ~26 分钟（22,002 episode，~35 ms/状态）
- C1 恢复轨迹：4 进程并行 ~60 分钟（22,002 任务，1.3-1.7 ep/s/进程）
- 恢复特征：4 进程 ~13 分钟
- 采样：两轮修正（① 等间隔强制端点 → bin 0 抬高 4×；② 改为随机相位步长，
  与池分布一致验证通过）
- 异常：HERE.parents 层级错误（第 4 次出现，已记录教训并加断言）；
  run_10b manifest.sha256 文件名硬编码（已改为派生）；target_emb 存了特征图
  而非描述子（加载时派生修复，语义与契约一致）
