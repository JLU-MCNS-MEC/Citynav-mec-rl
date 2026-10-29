# 实验 03：固定 50 m 与四动作教师

- 日期：2026-10-29
- 状态：**完成**（转换 + 回放 + 契约检查全部通过）
- 依赖：实验 01（固定高度约定）、实验 02（质量审计）
- 数据：Train-Seen 22,002 条；Val-Seen / Val-Unseen / Test-Unseen 未参与
- 脚本：`scripts/four_action_teacher.py`、`scripts/convert_four_action_teacher.py`、
  `scripts/verify_teacher.py`、`scripts/test_teacher_contract.py`
- 结果：`results/summary.json`、`results/teacher_episodes.csv`、
  `results/teacher_contract.json`、`results/sample_labels.npz`；
  完整标签（20.2 MB）在服务器 `results/four_action_labels.npz`

## 1. 方法

### 1.1 固定高度与清理

- 保留人工轨迹的水平坐标，高度统一为 `z = GROUND_LEVEL[map] + 50 m`（实验 01 的官方口径）。
- 删除相邻水平距离 <0.05 m 的重复点；**不保留也不使用旧的 GoUp/GoDown 标签**。
- 目标统一取 `target_positions[-1]`，STOP 半径 **20 m（水平）**，与官方成功判据一致；
  marker 只作为路径终点/地标，不替代目标。

### 1.2 四动作教师

按官方 `LookaheadTeacherParams(lookahead=1)` 与 `interpolated_trajectory`：

1. 沿清理后的路径按 >5 m 间隔重采样 waypoints，末尾追加 marker（若距上一个 waypoint >5 m）；
2. 从起点位姿（yaw 取首点方向向量）开始，每步取**全局最近 waypoint**，看向下一个点；
3. 航向差 >30° 转向 30°，否则前进 5 m；**没有升降动作**；
4. 与目标的水平距离 ≤20 m 时输出 STOP；否则继续，直到步数上限。

步数上限：`max(120, 2.5 × path_len / 5 + 100)`，封顶 2000。
若到达上限仍未进入 20 m，标记为 `step_limit` 失败；**不追加任何伪 STOP**。

## 2. 正确性检查（全部通过）

| 检查 | 结果 |
|---|---|
| 动作 ID 合法（仅 0–3） | ✅ |
| 无 GoUp/GoDown | ✅ |
| states 与 actions 一一对齐 | ✅ |
| 成功 episode 以 STOP 结尾 | ✅ |
| 失败 episode 不含 STOP | ✅ |
| STOP 时距离 ≤20 m | ✅（最大 **19.9925 m**） |
| 从起点用动作序列重放与记录 states 一致 | ✅（最大误差 **2.54e-4**，float32 量化） |
| 失败原因取值合法 | ✅ |

单元测试：`scripts/test_teacher_contract.py` **8/8 通过**（前进恰好 5 m、转向恰好 30°、
20 m 停止、转向后前进、失败无 STOP、waypoint >5 m、重复点清理）。

## 3. 转换结果

| 指标 | 值 |
|---|---:|
| episode 数 | 22,002 |
| 回放成功 | **19,105（86.83%）** |
| 失败 | 2,897 |
| ├ 步数上限仍未到达 | 2,331（10.6%） |
| ├ 人工路径从未进入 20 m | 565（2.6%） |
| └ 退化轨迹 | 1 |
| 状态步数（全部） | 2,571,989 |
| 状态步数（成功） | 1,342,094 |

成功回放的步数分布：

| 指标 | 值 |
|---|---:|
| 均值 | 70.2 |
| 中位 | 60 |
| p95 | 165 |
| p99 | 269 |
| 最大 | 1,077 |

（这些数字将用于实验 06 的闭环步数上限设定，而不是沿用旧的 100 步。）

### 3.1 动作分布（仅成功 episode，1,342,094 步）

| 动作 | 次数 | 每 episode 平均 |
|---|---:|---:|
| Forward | 988,147 | 51.7 |
| TurnLeft | 166,987 | 8.74 |
| TurnRight | 167,855 | 8.79 |
| Stop | 19,105 | 1.00（每条恰好一次） |

左右转几乎完全平衡；没有任何升降动作。

### 3.2 逐地图成功率

24 张训练地图的成功率范围 **78.9%–100%**：

| 最低 5 张 | 成功率 |
|---|---:|
| cambridge_block_19 | 78.9% |
| cambridge_block_7 | 82.0% |
| cambridge_block_13 | 82.6% |
| cambridge_block_9 | 82.9% |
| cambridge_block_28 | 83.3% |

最高的 `birmingham_block_11`（17 条）与 `birmingham_block_13`（30 条）接近 100%，
但样本很少。逐地图完整表在 `results/summary.json` 的 `by_map`。

### 3.3 与实验 02 的差异

实验 02 的回放成功 19,059 条，本阶段 19,105 条（+46）。原因是本阶段把步数上限从
`1.6×path/5+100` 放宽到 `2.5×path/5+100`；其余转换与重放逻辑一致。
这 46 条属于“教师需要更多步数才能绕回”的案例，不是标签差异。

## 4. 失败分析

- **步数上限 2,331 条**：30° 离散转向 + 5 m 前进无法在预算内跟随某些来回绕行/急转路径。
  这些 episode 的教师轨迹与目标越走越远，属于几何不可重放，**不是人工路径错误**。
- **从未进入 20 m 565 条**：人工路径本身没有到达成功半径，无法生成成功 STOP 标签。
- 1 条退化轨迹（点数不足）。
- 所有失败都保留原始轨迹 ID 与原因，写入 `results/teacher_episodes.csv`；
  训练集只会使用成功回放的 19,105 条（与实验 02 的“高质量”集合基本一致）。

## 5. 产物

| 文件 | 说明 | 是否入库 |
|---|---|---|
| `results/four_action_labels.npz` | 全部 22,002 条：states、actions、waypoints、目标、元数据 | 否（服务器，20.2 MB） |
| `results/sample_labels.npz` | 前 200 条的同样结构，供复核 | 是（197 KB） |
| `results/teacher_episodes.csv` | 逐 episode：成功/原因/步数/距离 | 是（4.3 MB） |
| `results/summary.json` | 统计汇总 | 是 |
| `results/teacher_contract.json` | 8 项契约检查结果 | 是 |

标签格式（npz）：

```
keys, maps, success, steps, reasons,
start_xy, start_yaw, target_xy, marker_xy, fixed_z_m,
states (N,3: x,y,yaw) + state_offsets,
actions (N,) + state_offsets,
waypoints (M,2) + way_offsets
```

## 6. 结论

1. 固定 50 m + 四动作的转换在 **86.8%** 的 Train-Seen episode 上可以被教师完整重放，
   并通过全部 8 项契约检查；可以进入实验 04/05。
2. 失败全部被显式标记（步数上限 / 从未进入 20 m / 退化），**没有伪造 STOP**。
3. 动作空间干净：只有 STOP、Forward、TurnLeft、TurnRight；左右转平衡。
4. 成功回放的步数 p99 = 269，最大 1,077；实验 06 的闭环上限应据此设定。

## 7. 限制

- 教师是 30°/5 m 的离散近似；2,331 条“不可重放”不代表人工轨迹不可达。
- 只处理 Train-Seen；验证划分保持完整，未做任何筛选。
- 固定高度使用官方 `GROUND_LEVEL`，不随地形起伏调整（符合任务定义）。

## 8. 复现

```bash
python experiments/03_fixed50m_four_action_teacher/scripts/convert_four_action_teacher.py --resume
python experiments/03_fixed50m_four_action_teacher/scripts/verify_teacher.py
python -m pytest experiments/03_fixed50m_four_action_teacher/scripts/test_teacher_contract.py
```

转换约 90 秒（单进程，分块 checkpoint 支持断点续跑）。

## 9. 下一步

实验 04：视觉目标与全局地图输入管线（俯视局部 RGB 编码、目标参考图像、16 步因果历史、
全局候选与目标信念），先做小规模输入/因果性验证，不构建融合 Token 缓存。
