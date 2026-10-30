# 固定 50 m 俯视视觉目标导航 — 阶段状态

- 分支：`exp/citynav-mec-fixed50m-topdown`
- 起点：`ebe557f`（旧版 `archive/citynav-token-v1-20261029`）
- 最近更新：2026-10-30 11:05（实验 05 正式训练完成：Transformer-M / seed 42 / 30 epochs）
- 训练种子：**只训练 seed 42**，seed 43 不再执行
- 任务定义：UAV 固定离地 50 m；输入为俯视局部 RGB、目标参考图像、16 步因果历史、
  全局俯视地图/地标候选与目标信念；**无语言输入、无 BERT、无网络状态**；
  动作四类（STOP / Forward 5 m / TurnLeft 30° / TurnRight 30°）；只训练 Transformer-M
  （seed 42）；不使用类别加权。评测器可用真实目标坐标算 SR，但不把它输入模型。
- `test_unseen` 保持封存，任何阶段不读取。

## 阶段状态

| 阶段 | 名称 | 状态 | 报告 | commit |
|---|---|---|---|---|
| 00 | 旧版归档与 Token 清理 | **完成**：2026-10-29 21:25:05 删除 13.49 GB，残留 0 | `00_archive_and_token_cleanup/REPORT.md`、`RECOVERY_2026-10-30.md` | `86d75ca`、`0f4f721` |
| 01 | 俯视地图资产审计 | **完成**：34 张栅格、28 图 PLY、GROUND_LEVEL 全覆盖 | `01_topdown_asset_audit/REPORT.md` | `fa31e99` |
| 02 | 人工轨迹质量审计 | **完成**：高 14,237 / 中 4,284 / 低 3,481；四动作回放成功 19,059 | `02_trajectory_quality_audit/REPORT.md` | `12f2d89` |
| 03 | 固定 50 m 与四动作教师 | **完成**：19,105/22,002 可重放，8 项契约全过 | `03_fixed50m_four_action_teacher/REPORT.md` | `0664591` |
| 04 | 视觉目标与全局地图输入管线 | **完成（修正）**：空间特征 7×7×2048、高程有效掩码、Recall@1/5/10 已修正；48 m 与 24 m 网格双划分全量验证 | `04_visual_goal_global_map_pipeline/REPORT.md`（旧版）+ `ADDENDUM_2026-10-30.md` | `722b62b`、`f058b82` |
| 03B | 直接目标导向教师 | **完成**：22,002/22,002 成功；修复 15°+30° 极限环（转向占比 32.3%→5.9%）；11 项契约通过 | `03b_target_directed_teacher/REPORT.md` | `0f79b22` |
| 05 | Transformer-M 正式训练（seed 42） | **完成**：30/30 epoch，79.97 min；best Val-Seen Macro-F1 **0.784**（epoch 1）；全量诊断 Val-Seen acc 0.934 / Macro-F1 0.762 / STOP 0.485、Val-Unseen acc 0.936 / Macro-F1 0.747；**STOP 是瓶颈**。09:02 的 candidates48 运行已判 INVALID | `05_transformer_m_training/REPORT.md` | `8b35c3b` |
| 06 | 闭环 SR 验证 | **未开始**；2026-10-29 的 3 条初步 episode **已作废**，见 `06_closed_loop_success_rate/INVALID_PRELIMINARY.md` | — | — |
| 07 | 采样/覆盖未解决问题 | **OPEN**：已建立归档文件并写入 03B 的覆盖数据，问题未解决 | `07_sampling_coverage_open_issue/OPEN_ISSUE.md` | 待填 |

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
  高 60.13 m，`cambridge_block_26` 高 52.65 m（不在开发划分内）。按 1 m 高程障碍栅格统计，
  28 张开发地图中只有 **5 个 1 m 单元** 高过 50 m（`cambridge_block_2` 3 个、
  `birmingham_block_4` 2 个），因此固定 50 m 事实上越过几乎所有建筑；
  实验 03B 仍按协议保留高程障碍图 + A*，全量只有 1 条任务触发绕行。
- 实验 03B（直接目标导向教师）：三个划分 **100% 成功**（22,002 / 2,498 / 2,826），
  每条任务恰好 1 个 STOP 且距离 ≤ 20 m；步数 均值 39.8 / 中位 39 / p95 74 / p99 85 / 最大 127；
  动作占比 Forward 91.6%、TurnLeft 2.95%、TurnRight 2.95%、Stop 2.5%。
- 03B 发现的真实缺陷：**15° 阈值 + 30° 离散转向会形成 L-F-R-F 二步极限环**
  （字面规则下转向占 32.3%、每任务 17.9 次、59.3% 的任务左右转次数几乎相等）。
  改成“15° 对齐、已对齐后 25° 才重新触发”的双阈值后，转向占比降到 5.9%，
  步数 55.3→39.8，航程开销 2.4%。旧产物完整保留在 `results_naive_15deg/`（服务器）。
- 旧 Token 数据集共 13.49 GB（核心 10.63 GB + 审计派生 2.91 GB），已于 21:25 删除，残留 0。
- `mec-v8.2` 无本地 Token 数据集；全盘无 `token_dataset_*` 目录。
- 实验 05（seed 42，30 epochs，candidates24，普通 CE）：
  - 训练 22,002 条 / 876,398 状态；30/30 epoch，墙钟 79.97 min，平均 159 s/epoch。
  - 训练损失 0.193→0.100 单调下降，梯度范数 0.51–0.67（裁剪阈值 1.0 从未触发），无 NaN/OOM。
  - best（按 Val-Seen Macro-F1）为 **epoch 1**：val acc 0.9426、Macro-F1 0.784、
    STOP 0.617、转向 0.742/0.671。按 acc 选会选 epoch 5（0.9461），但其 STOP 只有 0.35。
  - 全量教师强制诊断（best.pt）：Val-Seen acc 0.934 / Macro-F1 0.762 / STOP 0.485；
    Val-Unseen acc 0.936 / Macro-F1 0.747 → 跨地图泛化与已见地图基本一致。
  - 混淆结构：STOP 有 47–48% 被判成 Forward，同时 34/1747 个 Forward 被误判 Stop；
    转向错误几乎全部是“转成 Forward”，左右转之间混淆仅 0.01。
  - 归一化改为单一路径（uint8→GPU），缓存与训练特征 `max_abs_diff = 0`；
    缓存为 episode 级 npz + journal（大小/SHA-256），全量校验 27,326 条 0 bad。

## 已核实的历史差异（不是数据错误）

- 实验 02 的教师回放成功数 **19,059**，实验 03 为 **19,105**（+46）。
  原因：实验 02 的步数上限是 `max(60, 1.6×path_len/5+100)`，实验 03 放宽为
  `max(120, 2.5×path_len/5+100)`；其余转换逻辑一致。实验 03 报告 §3.3 有原始说明。
- 实验 02 已由旧分支于 2026-10-29 22:35:56 正常收尾，无残留进程。

## 未解决问题

- **STOP 是实验 05 暴露的主要瓶颈**：类别极不平衡（Forward 91.6% / Turn 5.9% / Stop 2.5%）
  且协议禁止加权，普通 CE 下 STOP 召回在 epoch 间 0.20–0.83 震荡、后段稳定在 0.46 左右，
  全量诊断 0.485。转向 F1 0.78–0.83 可用。Stage 06 闭环必须重点测 STOP；
  在此之前不得用类别加权“修”这个结果。
- 实验 05 的 best checkpoint 出现在 epoch 1、val acc 在 epoch 5 见顶，之后 25 个 epoch
  主要是过拟合（末 epoch train 0.963 / val 0.937）；这是固定 30 epoch 协议的产物，
  没有早停。
- `nodata` 区域在 03B 与 04 中都按“无数据、可飞/置 0”处理；若 nodata 中藏有未测绘建筑，
  两处会同时忽略它。这是一个共同假设，尚未被独立数据验证。
- 03B 的 A* 分支只在 1 条任务上生效，避障能力没有被充分检验。
- 官方 CityNav 仓库未在本机；`/tmp/citynav-official/` 的 `mapdata.py` 可用，
  但 `rasterize.py` / `cropclient.py` / `mturk_trajectory.py` 仍需核对（实验 01 已按 TIF 实测口径处理）。
- 目标参考图像只能提供图像内容，不得提供真实坐标/ID；全局定位靠视觉候选匹配。
- 采样/覆盖问题（各图、各长度、各阶段动作分布、训练/闭环分布偏移）尚未系统归档，见阶段 07。
