# 数据审计与六动作标签分布

- 日期：2026-10-27
- 命令：`python scripts/audit_dataset.py --splits train_seen val_seen val_unseen --teacher-sample 100`
- 未读取 `test_unseen`

## 1. 官方划分与资产

| split | 记录数 | 预期 | 地图数 | 描述严格匹配 | 描述归一化匹配 | 目标越界 | 重复 episode id |
|---|---:|---:|---:|---:|---:|---:|---:|
| train_seen | 22,002 | 22,002 | 24 | 21,463 | 22,002 | 0 | 0 |
| val_seen | 2,498 | 2,498 | 24 | 2,430 | 2,498 | 0 | 0 |
| val_unseen | 2,826 | 2,826 | 4 | 2,757 | 2,826 | 0 | 0 |

- CityRefer：34 个 block、26,274 个对象；
- 三个开发划分引用 28 个 block，全部具备 PNG/TIF/PLY；
- PNG/TIF 尺寸逐图一致，地理变换均为 0.1 m；尺寸随 block 裁剪变化；
- 描述严格差异全部来自空白和 ASCII 双引号；归一化后 100%；
- 目标位置与 CityRefer 对象 `position` 对齐（target-object 距离 0）。

## 2. 教师语义

- 六元组：`x,y,z,dx,dy,dz` → `x,y,z,yaw,pitch`；
- 真实轨迹高度保留，不再固定 50 m；
- 教师路径：原始轨迹 + `marker_positions[-1]`，按官方 5 m 间隔插值；
- 教师停止阈值：5 m，针对 marker；
- 评估成功半径：20 m，针对 `target_positions[-1]`；
- lookahead=1（官方 episode 生成配置），30° 转向阈值，1 m 垂直阈值；
- 动作执行：5 m 前进 / 30° 右左转 / 2 m 升降。

## 3. 六动作计数（教师抽样 100 条/划分）

| split | Stop | MoveForward | TurnRight | TurnLeft | GoUp | GoDown | 成功率（target<20m） |
|---|---:|---:|---:|---:|---:|---:|---:|
| train_seen | 100 | 6,654 | 1,096 | 992 | 3,617 | 8,044 | 0.87 |
| val_seen | 100 | 6,423 | 1,115 | 1,078 | 3,791 | 8,198 | 0.89 |
| val_unseen | 100 | 5,947 | 879 | 812 | 2,942 | 7,662 | 0.90 |

- 教师最终 marker 距离中位约 3.3–3.4 m，p90 约 4.7–5.0 m；
- 教师最终 target 距离中位约 7.2–7.5 m；部分 episode 因 marker 与 target
  相距较远，target 成功率低于 1；
- GoUp/GoDown 样本充足（开发划分抽样中 Up 2,942–3,791，Down 7,662–8,198），
  不需要为垂直动作单独过采样；
- 六动作标签与五类移动头映射：Forward/Right/Left/Up/Down → head 0..4，
  Stop 由 Arrival 头负责。

## 4. 说明

审计只读取 JSON / CityRefer / PLY 头部/少量顶点；运行前后原始数据快照一致
（摘要 `2c784870be858d7d801ee7c12e68c81b44b11cb7`）。完整审计 JSON/Markdown
位于本地 `reports/`，不提交到 GitHub。
