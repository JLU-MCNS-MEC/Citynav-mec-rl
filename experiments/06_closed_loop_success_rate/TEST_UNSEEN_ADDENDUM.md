# Test-Unseen 追加闭环评测（2026-10-30）

> 用户在 Stage 07 诊断完成后指示 **Test-Unseen 可以不封存**。
> 本附录只做**已训练模型（Stage 05 best.pt）的闭环评测**：
> 没有重新训练、没有调参、没有生成 03B 教师标签、没有修改模型或输入规则。
> 原有的 `REPORT.md` 与历史结果保持不变，本文件是追加。

## 1. 协议

- 模型：`runs_v2/seed42/best.pt`，SHA-256 `077f01a4…c0ce2`（与 Stage 06 相同，未重选）
- 任务：官方 `citynav_test_unseen.json` 全部 **5,311** 条，6 张地图
  （birmingham_block_0/6、cambridge_block_8/14/26/32）
- 起点/目标解析：复用 03B 的 `direct_teacher.start_pose`（起点 xy、首点航向）与
  `target_positions[-1][:2]`，与验证任务的定义一致
- 动作/判据：与 Stage 06 完全相同（四动作、200 步、STOP 且水平距离 ≤ 20 m）
- 候选库：开发集 `cache_v2/candidates24` 不包含这 6 张地图，按**完全相同的参数**
  （stride 24 m、patch 96 m、min-valid 0.7、min-coverage 0.4、同一冻结 ResNet）
  补建到 `cache_v2_test/candidates24`，共 **829 个候选**；开发缓存未改动
- 分组：无 03B 教师标签 → **没有长度分组、没有 SPL**；改用起始距离、地图、cohort 分层
- Cohort A/B：seed 20261030 对整个 split 确定性打乱后交替划分（A 2,656 / B 2,655）

## 2. 主结果

| 范围 | n | SR | 95% CI | OSR | STOP 率 | NE 中位 | 停止误差中位 | 步数均值 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| **Test-Unseen** | **5,311** | **25.0%** | [23.9, 26.2] | 29.8% | 56.7% | 32.8 m | 21.3 m | 45.4 |
| cohort A | 2,656 | 26.5% | [24.9, 28.2] | 31.4% | 57.1% | 31.7 m | 20.9 m | 44.7 |
| cohort B | 2,655 | 23.6% | [22.0, 25.2] | 28.1% | 56.3% | 33.6 m | 21.7 m | 46.1 |

运行：4 worker，**16.4 分钟**（5,311 条，0.19 s/条）。

## 3. 按地图

| 地图 | n | SR | OSR | 说明 |
|---|---:|---:|---:|---|
| birmingham_block_0 | 573 | 26.7% | 35.4% | |
| birmingham_block_6 | 176 | 18.2% | 30.1% | 任务少 |
| cambridge_block_14 | 832 | 26.8% | 30.6% | |
| cambridge_block_26 | 2,135 | 25.8% | 31.0% | |
| **cambridge_block_32** | **220** | **7.7%** | 8.2% | **候选库为空（0 个候选）** |
| cambridge_block_8 | 1,375 | 25.7% | 28.4% | |

`cambridge_block_32` 是极端但信息量最大的对照：该地图没有任何满足
（min-valid 0.7 / min-coverage 0.4）的候选，`belief_features` 返回全零 12 维向量。
220 条任务里 **200 条在约 2 步内输出 STOP**（`premature_stop`），SR 仅 7.7%。
这与 Stage 07 审计的判断一致：**信念是模型的主要定位信号**，信念缺失时策略直接退化为"停"。

## 4. 按起始距离

| 起始距离 | n | SR | 越界率 |
|---|---:|---:|---:|
| 0–20 m | 60 | 95.0% | 0.0% |
| 20–50 m | 321 | 42.1% | 1.6% |
| 50–100 m | 826 | 40.7% | 6.5% |
| 100–200 m | 1,854 | 26.7% | 32.4% |
| >200 m | 2,250 | 13.6% | 62.7% |

与 Stage 06 的形状一致：起始距离越远，越容易在到达前越界。

## 5. 失败模式与动作

| 失败类别 | 条数 | 比例 |
|---|---:|---:|
| 成功 | 1,330 | 25.0% |
| 过早 STOP | 1,658 | 31.2% |
| 进入 20 m 未停 | 251 | 4.7% |
| 越界 | 2,071 | 39.0% |
| 碰撞 | 1 | 0.02% |
| 到 200 步上限 | 0 | 0 |

执行动作：Forward **96.1%**、TurnLeft 1.40%、TurnRight 1.28%、Stop 1.25%。
与 Val-Seen/Val-Unseen 的失败构成相同：越界 + 过早 STOP 占 70%，转向依然稀少。

## 6. 目标图像有效性

- `goal_image_invalid`：0 条
- 有效比例 < 0.5：**97 条，SR 5.2%**（≥0.5 的子集 SR 25.4%）
- 与 Stage 06 的低有效目标现象一致。

## 7. 与 Val-Seen / Val-Unseen 的对比

| | Val-Seen | Val-Unseen | **Test-Unseen** |
|---|---:|---:|---:|
| n | 2,498 | 2,826 | 5,311 |
| 地图 | 24（训练见过） | 4（训练未见） | **6（训练未见）** |
| SR | 24.9% | 22.5% | **25.0%** |
| OSR | 30.7% | 28.7% | 29.8% |
| STOP 率 | 53.6% | 49.2% | 56.7% |

Test-Unseen 的 SR 与 Val-Seen 基本持平、略高于 Val-Unseen。解释（有证据的推断）：
Test 的起始距离略近（中位 176.3 m；0–50 m 占 **7.2%**，Val-Seen 5.5%、Val-Unseen 4.0%），
且 6 张图里有 5 张候选库正常；`cambridge_block_32` 的 220 条把整体拉低约 0.7 个百分点。
**这不是"新地图泛化更好"的证据**——任务构成不同（起始距离分布、地图组合），不能直接下结论；
唯一能说的是：在新地图上 SR 没有崩溃，量级与验证集一致。

## 8. 限制

- **没有长度分组与 SPL**：Test-Unseen 没有 03B 教师标签；本报告的 SR 不与 Stage 06 的
  Short/Medium/Long 表直接对齐。
- Cohort A/B 相差 **2.9 个百分点**（Val 上 ≤1.1 pp），大于抽样误差（约 ±1.1 pp）；
  原因是 cohort 是对整个 split 打乱后交替划分，没有按地图分层。**A/B 不是配对比较**。
- `cambridge_block_32` 的候选库为空是数据限制（该图没有满足条件的候选），不是评测 bug；
  它的 220 条任务保留了，没有排除。
- 与官方 CityNav 的 SR 仍不可直接比较（官方是语言目标 + 六动作 + 3D 20 m，见
  `07_sampling_coverage_open_issue/LITERATURE_PROTOCOL_COMPARISON.md`）。

## 9. 结论

- Test-Unseen 闭环 **SR 25.0% [23.9, 26.2]**，与 Val-Seen 24.9% / Val-Unseen 22.5% 同一水平。
- 失败构成与验证集一致：越界 39.0%、过早 STOP 31.2%，转向仅 2.7%。
- `cambridge_block_32`（零候选、全零信念）的 220 条任务 SR 7.7%、200 条两步内 STOP，
  是"信念主导策略"的又一个直接证据。
- 本阶段没有训练、调参或模型修改；Test-Unseen 现已解封，但**未参与任何训练或模型选择**。

## 10. 视频案例（真实逐步记录）

从 5,311 条 Test-Unseen 记录中选 **6 条成功 + 5 条失败**，渲染方式与 Stage 06 视频相同
（左：真实 North-up 地图 + 20 m 圈 + 轨迹/航向；右：该步真实 `heading_crop`；底部：step、动作、
距离、四类概率、结果）。H.264 1280×720、10 fps、每步 3 帧，**离散动作回放**。

**成功（每张未见地图 1 条）**

| episode | 地图 | 步数 | 说明 |
|---|---|---:|---|
| test_unseen:00318 | birmingham_block_0 | 20 | 转弯最多的成功 |
| test_unseen:00567 | birmingham_block_6 | 17 | |
| test_unseen:01128 | cambridge_block_14 | 21 | |
| test_unseen:01828 | cambridge_block_26 | 11 | |
| test_unseen:03230 | cambridge_block_32 | **1** | 零候选、全零信念；起点已在 20 m 内，立即 STOP（该图 17 条成功全部是 1 步） |
| test_unseen:03663 | cambridge_block_8 | 19 | |

**失败**

| episode | 类型 | 地图 | 步数 | 说明 |
|---|---|---|---:|---|
| test_unseen:00002 | premature_stop | birmingham_block_0 | 23 | 停在 27.3 m |
| test_unseen:03228 | zero_belief_stop | cambridge_block_32 | 1 | 无候选 → 全零信念 → 立即 STOP |
| test_unseen:00000 | out_of_bounds | birmingham_block_0 | 56 | 直飞出界 |
| test_unseen:00008 | reached_but_not_stopped | birmingham_block_0 | 15 | 进入 20 m 后继续飞走 |
| test_unseen:02536 | collision | cambridge_block_26 | 28 | 唯一一次碰撞 |

- 索引：[`videos/test_unseen/VIDEO_INDEX.md`](videos/test_unseen/VIDEO_INDEX.md)
- 成功汇总片：[`success_compilation.mp4`](videos/test_unseen/success_compilation.mp4)（6 条，3.2 MB）
- 封面拼图：[`video_montage.png`](videos/test_unseen/video_montage.png)
- 清单（含每个 MP4 的 SHA-256）：[`video_manifest.csv`](videos/test_unseen/video_manifest.csv)
- 全部视频合计约 9 MB。

## 11. 复现

```bash
# 1) 任务清单（5,311 条，起点/目标解析与 03B 一致）
python -I -u experiments/06_closed_loop_success_rate/scripts/build_test_manifest.py

# 2) 6 张 Test 地图的 24 m 候选库（与开发集同参数）
python -I -u experiments/06_closed_loop_success_rate/scripts/build_test_candidates.py

# 3) 闭环评测（tmux citynav06_test，约 16 分钟）
python -I -u experiments/06_closed_loop_success_rate/scripts/evaluate_checkpoint.py \
  --manifest results/test_unseen/manifest.json --out-dir results/test_unseen \
  --split test_unseen --workers 4 \
  --candidates ../04_visual_goal_global_map_pipeline/cache_v2_test/candidates24

# 4) 汇总
python -I -u experiments/06_closed_loop_success_rate/scripts/summarize_test_unseen.py
```

结果：`results/test_unseen/summary.json`、`summary_tables.csv`、`manifest.sha256 = 0efcee1d…9b4c`；
逐条记录在服务器 `results/test_unseen/episodes/`（不入库）。
