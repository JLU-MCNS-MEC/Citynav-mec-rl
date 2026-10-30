# 数据血缘审计（Stage 00–06）

> 本文件只记录**已经核对过源文件与代码**的事实；推断一律标注"推断"，无法验证的标注"未验证"。
> 所有路径相对仓库根目录。

## 1. 一眼看清

```
DATA/rgbd/*.png + *.tif            (34 张地图；俯视 RGB + 高程，0.1 m/px)
        │
        ├─ 01 资产审计 ── 28 张开发地图、GROUND_LEVEL、PLY(仅资产核对)
        │
        └─ 原始人工轨迹 (CityNav 数据集)
                 │
                 ├─ 02 质量审计：22,002 条打分（高 14,237 / 中 4,284 / 低 3,481）
                 │        └─ 只产出分析标签；**没有**用于最终训练筛选
                 │
                 ├─ 03 人工路径跟随教师：22,002 条 → 19,105 条可重放（86.83%）
                 │        └─ 四动作是对人工路径的**投影**；已被 03B 取代，不进入 05
                 │
                 └─ 03B 直接目标导向教师：22,002 / 2,498 / 2,826 全部成功
                          │  只继承：起点 xy、起点航向、最终目标 xy
                          │  重新规划：直线或高程 A*，双阈值防极限环
                          │
                          ├─ 04 视觉目标管线：7×7×2048 空间特征、2048 目标嵌入、
                          │      16 步因果历史、12 维目标信念；候选库 48 m → 24 m
                          │
                          ├─ 05 训练：22,002 episode / 876,398 决策状态；seed 42；30 epoch
                          │      → best.pt（epoch 1，Val-Seen Macro-F1 0.7844）
                          │
                          └─ 06 闭环：Val-Seen 2,498 + Val-Unseen 2,826 = 5,324 条
                                 → SR 23.7%
```

## 2. 分阶段输入 / 输出 / 数量

| 阶段 | 输入 | 处理 | 输出 | episode / 样本 | 地图 |
|---|---|---|---|---|---|
| 00 | 旧分支 Token 数据 13.49 GB | 归档并删除 | 清理记录 | — | — |
| 01 | `DATA/rgbd/` 34 张 PNG+TIF | 资产审计、固定高度约定 | 审计表、GROUND_LEVEL | 34 栅格 / 28 开发地图 | 28 |
| 02 | Train-Seen 人工轨迹 | 50 m 投影 + 质量评分 | 高/中/低标签 | 22,002（14,237/4,284/3,481） | 24 |
| 03（旧） | 02 人工轨迹 | **路径跟随投影**为四动作 | `four_action_labels.npz` | 19,105 可重放 | 24 |
| 03B（现行） | 02 的起点/航向 + 最终目标 | **重新规划**（直线 / A*） | `teacher_{split}.npz/csv` | 22,002 / 2,498 / 2,826 | 24 / 24 / 4 |
| 04 | 01 栅格 + 03B 目标 | 裁剪、ResNet 特征、候选匹配 | `cache_v2/candidates24`、RECALL 报告 | 2,355 候选（24 m 网格） | 28 |
| 05 | 03B + 04 特征 | Transformer-M 训练 | `runs_v2/seed42/best.pt` | **22,002 episode / 876,398 状态** | 24 |
| 06 | 05 best.pt + 03B 验证任务 | 闭环 rollout | 5,324 条逐步记录 | 5,324 episode | 28 |

## 3. 七个必须核实的问题

### 3.1 Stage 02 的质量评分是否参与正式训练数据筛选？

**没有。** 证据：

- 02 把 22,002 条分成高 14,237 / 中 4,284 / 低 3,481，但只写进分析报告（`02_trajectory_quality_audit/REPORT.md`），
  没有生成训练用白名单。
- 03 的四动作重放对 22,002 条全部尝试，成功 19,105；报告明确写"训练集只会使用成功回放的 19,105 条"。
- **但 03 已被 03B 取代**：03B 对 22,002 条全部重新规划，**100% 成功**，Stage 05 的训练缓存
  `features_v2/train_seen/index.json` 记录 `episodes_total = 22002`，即**全部 22,002 条都进入了训练**。
- 结论：02 的质量分只影响旧的 03 时代；**现行 05 训练没有按质量筛选**，低质量人工轨迹的起点与目标同样在训练集里。

### 3.2 Stage 03 的人工路径跟随标签是否被 03B 取代？

**是。** 03B 报告首页写明"保留旧实验 03，本阶段不覆盖、不替代"；但 05 的 `dataset_v2.py` 只读
`03b_target_directed_teacher/results/teacher_{split}.npz`，`03_fixed50m_four_action_teacher/` 的标签
**没有任何下游引用**（grep `four_action_labels` 在 04–06 脚本中为 0 次）。

### 3.3 03B 是否只继承起点、起始航向与最终目标？

**是。** `03b_target_directed_teacher/scripts/run_teacher.py` 的文档字符串写明：
"Human trajectories are used for the start xy/yaw and for the offline human path length"。
`direct_teacher.py::plan_path(start_xy, goal_xy)` 只用这两个点做直线 / A* 规划。
03B 报告 §1 给出契约测试 `test_human_path_middle_points_and_marker_do_not_change_labels`：
删除中间点、替换 marker、替换语言描述后动作与状态序列逐位相同。
人工 `path_len_m` 只作为离线诊断量写入 CSV（用于 SPL），不进入模型。

### 3.4 六动作 → 四动作：映射、投影还是重新规划？

两者都有，且必须区分：

- **03（旧）**：把人工轨迹**投影**到四动作——航向差 >30° 就转 30°，否则前进 5 m，末点 STOP。
  人工路径的中间点仍然决定了轨迹形状，因此这是"路径跟随的重参数化"。
- **03B（现行）**：**重新规划**——只看起点与目标，直线可达就走直线，否则在 1 m 高程障碍栅格上 A*。
  中间路径不继承。这是新生成的轨迹，**不是**对人工轨迹的动作映射。

因此本仓库里"直接教师轨迹"与"人工轨迹"是两种数据；任何报告都不应把前者称为后者。

### 3.5 固定高度与净空

- 高度定义：`z = GROUND_LEVEL[map] + 50 m`（`03b/obstacles.py:ALTITUDE_M`，`pipeline.py::load_ground`）。
- `GROUND_LEVEL` 是**每张地图一个常数**，不是逐点地形。因此"离地 50 m"是相对该地图参考地面的
  巡航高度，**实际净空 = 50 − (局部地形高程 − GROUND_LEVEL)**，随地形起伏变化。
- 障碍判据：1 m 单元内最大高程 ≥ `GROUND_LEVEL + 49.5 m` 视为阻挡。
- 结果（03B 报告 §1.1）：28 张开发地图里只有 2 张存在超过飞行高度的结构，共 **5 个 1 m 单元**
  （`cambridge_block_2` 最高 68.19 m AGL 3 个；`birmingham_block_4` 60.13 m 2 个）。
  也就是说巡航高度低于这 5 个单元的建筑顶部，其余地方远高于建筑。
- `nodata`（高程标签 -9999 与填充平台）**按可飞处理**，这是 03B/04 的共同假设，尚未被独立数据验证。

### 3.6 PLY 的角色

- 03B/04/05/06 的脚本中对 `.ply` 的引用数为 **0**（`grep -rn "\.ply"` 全部命中为空）。
- 训练与闭环只读 `DATA/rgbd/<map>.png`（俯视 RGB）与 `<map>.tif`（高程）。
- PLY 只在实验 01 的资产审计（以及旧版第一视角渲染的归档）里出现，属于源数据/资产核对，
  **不参与**当前管线的任何输入。

### 3.7 03B 100% 成功率依赖真实目标坐标，对任务难度意味着什么？

- 是的：`plan_path` 的 goal 就是真实目标 xy，教师是"已知终点"的规划器，因此 22,002 条全部成功。
- 学生（Transformer-M）**不知道**目标坐标；它只有目标参考图像与由候选匹配得到的 12 维信念。
- 因此训练标签是"已知终点时的近最优路径"，学生要在一个**部分可观测**的问题里模仿它。
  这本身是合理的模仿学习设定，但意味着：
  - 教师轨迹的状态分布非常"干净"（目标几乎总在正前方，见 `CLOSED_LOOP_FAILURE_AUDIT.md` §2）；
  - 学生一旦偏离这个分布，训练集中几乎没有对应的纠正样本。
- 不能把"教师 100% 成功"当作任务难度或模型能力的证据。

## 4. 划分与地图

| 划分 | episode | 地图数 | 地图集合 | 用途 |
|---|---|---|---|---|
| Train-Seen | 22,002 | 24 | 24 张开发地图 | 训练 |
| Val-Seen | 2,498 | 24 | **与 Train-Seen 完全相同**（子集） | 验证 / 闭环评测 |
| Val-Unseen | 2,826 | 4 | `birmingham_block_5`、`cambridge_block_10`、`cambridge_block_2`、`cambridge_block_3` | 验证 / 闭环评测 |
| Test-Unseen | — | — | — | **全程未读取** |

已核实：Val-Seen 地图 ⊆ Train-Seen 地图；Val-Unseen 的 4 张与 Train-Seen / Val-Seen **零交集**。
因此 Seen/Unseen 的差别是**地图级**泛化，不是任务级。

## 5. 轨迹统计对照

| 指标 | 人工轨迹（02，22,002） | 03 路径跟随（19,105 成功） | 03B 直接教师（22,002） |
|---|---|---|---|
| 步数 均值 / 中位 / 最大 | — / 68 / — | — | **39.8 / 39 / 127** |
| 起点到目标 中位 | — | — | **189.6 m** |
| STOP 占比 | — | 1 次/条 | **2.51%** |
| Forward 占比 | — | — | **91.58%** |
| TurnLeft / TurnRight | — | — | **2.95% / 2.95%** |
| 成功率 | — | 86.83% 可重放 | **100%（规划器可知终点）** |

（人工轨迹的步数与动作分布在 02 报告中是"四动作回放"口径，与 03B 不可直接比较，故留空。）

## 6. 最终训练到底用了什么

- **episode**：22,002 条 Train-Seen（全部）；Val-Seen / Val-Unseen 不参与训练。
- **状态池**：876,398 个决策状态（= 22,002 条轨迹的全部状态）。
- **每 epoch 实际抽样**：每条 episode 8 个状态（不足 8 的按实际长度），共 173,107 次抽样/epoch，
  1,353 个 optimizer step/epoch；30 epoch 合计 5,193,210 次抽样、40,590 次梯度更新。
- **覆盖率**：866,106 / 876,398 = **98.83%** 的状态至少被抽到一次；STOP 状态覆盖 99.39%。
- 细节与分层覆盖见 `TRAINING_SAMPLING_AUDIT.md`。

## 7. 证据文件

- `02_trajectory_quality_audit/REPORT.md`
- `03_fixed50m_four_action_teacher/REPORT.md`
- `03b_target_directed_teacher/REPORT.md`、`scripts/run_teacher.py`、`scripts/direct_teacher.py`、`scripts/obstacles.py`
- `04_visual_goal_global_map_pipeline/REPORT.md`、`ADDENDUM_2026-10-30.md`
- `05_transformer_m_training/REPORT.md`、`scripts/dataset_v2.py`、`scripts/train_v2.py`
- `06_closed_loop_success_rate/REPORT.md`、`scripts/closed_loop.py`
- `05_transformer_m_training/features_v2/*/index.json`（缓存身份：candidates24 / episode-v1）
