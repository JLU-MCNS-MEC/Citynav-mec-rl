# 固定 50 m 俯视视觉目标导航 — 阶段状态

- 分支：`exp/citynav-mec-fixed50m-topdown`
- 起点：`ebe557f`（旧版 `archive/citynav-token-v1-20261029`）
- 最近更新：2026-10-30 07:30（中断恢复检查，见
  `00_archive_and_token_cleanup/RECOVERY_2026-10-30.md`）
- 任务定义：UAV 固定离地 50 m；输入为俯视局部 RGB、目标参考图像、16 步因果历史、
  全局俯视地图/地标候选与目标信念；**无语言输入、无 BERT、无网络状态**；
  动作四类（STOP / Forward 5 m / TurnLeft 30° / TurnRight 30°）；只训练 Transformer-M
  （seed 42/43）；不使用类别加权。评测器可用真实目标坐标算 SR，但不把它输入模型。
- `test_unseen` 保持封存，任何阶段不读取。

## 阶段状态

| 阶段 | 名称 | 状态 | 报告 | commit |
|---|---|---|---|---|
| 00 | 旧版归档与 Token 清理 | **完成**：2026-10-29 21:25:05 删除 13.49 GB，残留 0 | `00_archive_and_token_cleanup/REPORT.md`、`RECOVERY_2026-10-30.md` | `86d75ca`、`0f4f721` |
| 01 | 俯视地图资产审计 | **完成**：34 张栅格、28 图 PLY、GROUND_LEVEL 全覆盖 | `01_topdown_asset_audit/REPORT.md` | `fa31e99` |
| 02 | 人工轨迹质量审计 | **完成**：高 14,237 / 中 4,284 / 低 3,481；四动作回放成功 19,059 | `02_trajectory_quality_audit/REPORT.md` | `12f2d89` |
| 03 | 固定 50 m 与四动作教师 | **完成**：19,105/22,002 可重放，8 项契约全过 | `03_fixed50m_four_action_teacher/REPORT.md` | `0664591` |
| 04 | 视觉目标与全局地图输入管线 | **初版完成，修正中**：报告写 7×7×2048 但 `encode_episode()` 实际返回 2048 全局池化；高位掩码、Recall@10 有缺陷 | `04_visual_goal_global_map_pipeline/REPORT.md`（旧版）+ 修正 ADDENDUM | `722b62b` |
| 03B | 直接目标导向教师 | **进行中**（新阶段） | `03b_target_directed_teacher/REPORT.md` | — |
| 05 | Transformer-M 训练与小规模诊断 | **未开始**；2026-10-29 23:06 之前的初步产物**已作废**，见 `05_transformer_m_training/INVALID_PRELIMINARY.md` | — | — |
| 06 | 闭环 SR 验证 | **未开始**；2026-10-29 的 3 条初步 episode **已作废**，见 `06_closed_loop_success_rate/INVALID_PRELIMINARY.md` | — | — |
| 07 | 采样/覆盖未解决问题 | 未开始 | `07_sampling_coverage_open_issue/OPEN_ISSUE.md` | — |

## 已确认事实

- `DATA/rgbd/` 已覆盖 34 张地图的 North-up RGB PNG + 高程 TIF，共 3.59 GB，PNG/TIF 一一对应，
  分辨率统一 0.1 m/px，无 CRS（x/y 即米制地图坐标），28 张开发地图资产无缺失。
- 官方固定高度约定：`z = GROUND_LEVEL[map] + 50`；局部观测是以该高度为半边长的
  正方形（100 m × 100 m），按航向旋转到机头朝前。
- 28 张开发地图的 PLY 全部存在（共 28.32 GB）；6 张无 PLY 的地图不在开发划分内。
- 22,002 条 Train-Seen：96.2% 的人工路径进入目标 20 m，86.6% 可被四动作教师回放；
  高质量 64.7%、中等 19.5%、低 15.8%；长轨迹高质量占比仅 32.8%。
- 四动作教师转换：19,105/22,002（86.83%）成功重放，失败 2,897（步数上限 2,331、
  从未进入 20 m 565、退化 1）；契约检查 8/8 通过，无伪 STOP、无升降动作。
- 成功回放步数：均值 70.2、中位 60、p95 165、p99 269、最大 1,077；左右转次数基本相等。
- 视觉目标管线（初版，数值待修正后重算）：冻结 ResNet-50 空间特征 7×7×2048；
  28 图 1,238 个固定候选（48 m 网格、96 m 切片）；Val-Seen 上 top1/top5 24 m 命中 54.5%/75.8%；
  历史 16 步严格因果。**初版 Recall@10 实为 Recall@5，不能引用。**
- 高程 TIF 的 `nodata` 标签是 `-9999`，实测最小值也为 `-9999`（并非 `min_z-1`）；
  开发地图 nodata 比例为 20–30%，`cambridge_block_4` 达 74.7%。
- 建筑高度：`cambridge_block_2` 最高比 `GROUND_LEVEL` 高 68.19 m，`birmingham_block_4`
  高 60.13 m，`cambridge_block_26` 高 52.65 m —— **固定 50 m 不能越过全部建筑**，
  实验 03B 必须使用高程障碍图 + A*。
- 旧 Token 数据集共 13.49 GB（核心 10.63 GB + 审计派生 2.91 GB），已于 21:25 删除，残留 0。
- `mec-v8.2` 无本地 Token 数据集；全盘无 `token_dataset_*` 目录。

## 已核实的历史差异（不是数据错误）

- 实验 02 的教师回放成功数 **19,059**，实验 03 为 **19,105**（+46）。
  原因：实验 02 的步数上限是 `max(60, 1.6×path_len/5+100)`，实验 03 放宽为
  `max(120, 2.5×path_len/5+100)`；其余转换逻辑一致。实验 03 报告 §3.3 有原始说明。
- 实验 02 已由旧分支于 2026-10-29 22:35:56 正常收尾，无残留进程。

## 未解决问题

- 实验 04 的输入契约与验证数值需按修正报告重算（空间特征、掩码、Recall@K、双划分验证）。
- 实验 03B 直接目标导向教师尚未产出；实验 05 必须等 03B 与 04 修正完成后再启动。
- 官方 CityNav 仓库未在本机；`/tmp/citynav-official/` 的 `mapdata.py` 可用，
  但 `rasterize.py` / `cropclient.py` / `mturk_trajectory.py` 仍需核对（实验 01 已按 TIF 实测口径处理）。
- 目标参考图像只能提供图像内容，不得提供真实坐标/ID；全局定位靠视觉候选匹配。
- 采样/覆盖问题（各图、各长度、各阶段动作分布、训练/闭环分布偏移）尚未系统归档，见阶段 07。
