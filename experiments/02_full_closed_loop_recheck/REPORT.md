# 实验 02：全模型闭环复测（500 步上限）

- 日期：2026-10-29
- 状态：**完成**（`val_seen` 18:08，`val_unseen` 见 §12）；两轮均按 500 步协议跑完，结论见 §7 与 §12。
- 依赖：实验 01（步数上限）、实验 03（STOP 与数据流审计）
- 未读取：`test_unseen`
- 原始结果：`runs/val_seen_{small,transformer_m}_roundA/`；逐条实时记录见 `LIVE_RESULTS.md`

## 1. 假设与目的

此前闭环在 100 步上限下全部失败，无法区分「截断」与「策略失败」。本轮固定 **500 步**
（实验 01 的结论：覆盖官方过滤后的全部 episode），检验三件事：

1. **H1**：把上限从 100 提到 500 后，是否出现成功轨迹；
2. **H2**：`small:full` 与 `transformer_m:full` 在闭环中是否有可区分的差异；
3. **H3**：模型是否**主动**输出 STOP，而不是被评测器强制终止。

评测器不向 MEC 提供任何目标真值（`closed_loop_eval.py:6,125-129,186-213`），
500 步只是统一的评测预算。

## 2. 设置

| 项 | 值 |
|---|---|
| 视觉编码器 | `runs/visual_encoder_citynav/abl_20261028_e2_fov_pose_seed42/best_encoder.pt`（冻结，E2-FOV-Pose，seed 42） |
| MEC | `data_cache/t5_frozen_1000/mec/small_full/mec.pt`、`.../transformer_m_full/mec.pt` |
| 检索器 | `data_cache/t5_frozen_1000/retrieval/retriever.pt`（`--use-retrieval`） |
| Token | `data_cache/t5_frozen_1000/tokens/t5_tokens.json`；GAB 阈值 0.038807（取自 token 元数据） |
| 变体 | 仅 `full`（不使用 `map_free`） |
| 步数上限 | 500（按用户要求**只**评估 500，不再计算 100/200/300/400 切点） |
| 裁剪半径 | 120 m |
| 并发 | 每个模型 8 分片；同划分两模型并发（16 进程） |
| 选择 | `selection/val_seen_roundA_indices.json`：每划分 24 条，按全量教师动作数三分位取 short/medium/long 各 8 条，共 17 张地图，教师动作数 51–478 |

说明：分片按 `map_name` 排序后连续切分，因此每个分片的 3 条 episode 通常落在同一张地图上，
单个分片的单步延迟由该地图决定。

## 3. 数据规模与进度

| 划分 | 模型 | 完成/应完成 | 完成时间 |
|---|---|---:|---|
| val_seen | small | 24/24 | 18:08:41 |
| val_seen | transformer_m | 24/24 | 18:08:27 |
| val_unseen | small | 进行中 | — |
| val_unseen | transformer_m | 进行中 | — |

- suite 自 **14:56:22** 启动，`val_seen` 一对模型于 **18:08:41** 结束，墙钟 **3 h 12 min**。
  （14:09 与 14:20 两次启动因会话中断被终止，14:56 重新拉起；journal 机制保证已完成
  episode 不会被重跑。）
- 实测单步延迟 **0.82–13.40 s**（均值 5.50 s），差异几乎全部来自地图渲染成本；
  单条 500 步轨迹耗时约 7–112 min。
- 16 分片并发时显存占用约 90 GB / 97.9 GB（val_unseen 运行期间实测）。

## 4. 指标定义

| 指标 | 定义（`closed_loop_eval.py:191-232`） |
|---|---|
| steps | 实际执行步数；未 STOP 则等于上限 |
| success (SR) | 终点到目标的 **3D** 距离 ≤ 20 m（**不要求 STOP**） |
| stopped | 模型 argmax 输出 STOP |
| stopped_within_20m | 输出 STOP 且终点 ≤ 20 m（更严格口径） |
| NE | 终点到目标的 3D 距离 |
| min_dist | 轨迹上到目标的**水平**最小距离 |
| entered_20m | 任意一步 3D 距离 ≤ 20 m |
| left_after_entering_20m | 进入 20 m 后又被走出 |
| oracle_success | min_dist ≤ 20 m（即该轨迹"路过"过目标） |
| nDTW / SPL | 对参考轨迹的归一化 DTW；SPL 在失败时恒为 0 |

注意：本报告的 SR 口径是「终点距离 ≤ 20 m」，与「在 20 m 内 STOP」不同；两种口径下
本轮结果都是 0，见第 5 节。

## 5. 结果

### 5.1 汇总（val_seen，n=24）

| 指标 | small:full | transformer_m:full |
|---|---:|---:|
| SR（终点 ≤20 m） | **0/24 (0%)** | **0/24 (0%)** |
| stopped / stopped_within_20m | 0 / 0 | 0 / 0 |
| 跑满 500 步 | 24/24 | 24/24 |
| NE 均值 / 中位 / 范围 (m) | 2140 / 2153 / 1882–2338 | 2136 / 2135 / 1872–2368 |
| 起点距离 均值 / 中位 (m) | 220 / 201 | 220 / 201 |
| min_dist 均值 / 中位 / 最小 (m) | 82.2 / 63.6 / **3.3** | 82.2 / 63.6 / **3.3** |
| min_dist <20 / <50 / <100 m | 3 / 9 / 17 | 3 / 9 / 17 |
| entered_20m | 3/24 (12.5%) | 3/24 (12.5%) |
| 进入后又离开 | 3/3 | 3/3 |
| oracle_success | 3/24 (12.5%) | 3/24 (12.5%) |
| nDTW 均值 | 0.119 | 0.121 |
| SPL 均值 | 0.000 | 0.000 |

要点：**终点距离是起点的约 10 倍**（220 m → 2140 m）。模型不是原地不动，而是持续朝
一个方向走并越走越远。

### 5.2 逐条结果（500 步，`min dist` 为水平距离，`20m 内步数` 为 3D 距离 ≤20 m 的步数）

| # | map | idx | bin | teacher steps | NE small (m) | NE tm (m) | min dist (m) | 进入20m | 20m内步数 | 步数 | 单步延迟 (s) |
|---:|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | birmingham_block_1 | 253 | short | 111 | 2241.0 | 2226.0 | 193.9 | — | 0 | 500 | 6.91 |
| 2 | birmingham_block_10 | 762 | medium | 160 | 1881.6 | 1871.6 | 21.1 | — | 0 | 500 | 1.88 |
| 3 | birmingham_block_12 | 817 | short | 55 | 2255.1 | 2255.1 | 7.1 | Y | 13 | 500 | 1.59 |
| 4 | birmingham_block_3 | 344 | medium | 147 | 2183.7 | 2178.6 | 83.6 | — | 0 | 500 | 8.67 |
| 5 | birmingham_block_9 | 683 | medium | 206 | 1979.7 | 1974.8 | 210.7 | — | 0 | 500 | 6.25 |
| 6 | birmingham_block_9 | 656 | short | 91 | 2111.3 | 2101.3 | 64.3 | — | 0 | 500 | 3.91 |
| 7 | cambridge_block_12 | 1427 | medium | 126 | 2236.7 | 2226.7 | 159.7 | — | 0 | 500 | 11.46 |
| 8 | cambridge_block_13 | 1608 | long | 313 | 2289.3 | 2284.3 | 39.3 | — | 0 | 500 | 7.68 |
| 9 | cambridge_block_13 | 1511 | short | 113 | 2050.2 | 2050.2 | 49.4 | — | 0 | 500 | 3.66 |
| 10 | cambridge_block_18 | 1718 | medium | 207 | 2223.4 | 2223.4 | 3.3 | Y | 6 | 500 | 3.82 |
| 11 | cambridge_block_18 | 1693 | short | 103 | 2166.6 | 2156.6 | 84.5 | — | 0 | 500 | 3.96 |
| 12 | cambridge_block_20 | 1957 | long | 236 | 2026.7 | 2021.7 | 32.3 | — | 0 | 500 | 9.14 |
| 13 | cambridge_block_21 | 2147 | long | 238 | 2136.2 | 2111.1 | 76.6 | — | 0 | 500 | 10.86 |
| 14 | cambridge_block_23 | 2194 | short | 63 | 2238.8 | 2233.8 | 61.1 | — | 0 | 500 | 0.82 |
| 15 | cambridge_block_25 | 2222 | long | 228 | 2282.4 | 2367.8 | 20.9 | — | 0 | 500 | 1.64 |
| 16 | cambridge_block_33 | 2421 | long | 282 | 2039.7 | 2034.6 | 20.4 | — | 0 | 500 | 4.07 |
| 17 | cambridge_block_33 | 2409 | medium | 133 | 1972.2 | 1957.2 | 62.9 | — | 0 | 500 | 3.76 |
| 18 | cambridge_block_4 | 891 | long | 294 | 2267.2 | 2262.2 | 83.1 | — | 0 | 500 | 1.39 |
| 19 | cambridge_block_4 | 880 | medium | 129 | 2337.9 | 2348.0 | 102.7 | — | 0 | 500 | 1.62 |
| 20 | cambridge_block_6 | 906 | medium | 143 | 1990.0 | 1975.1 | 236.0 | — | 0 | 500 | 9.54 |
| 21 | cambridge_block_6 | 913 | short | 98 | 2138.9 | 2114.0 | 148.5 | — | 0 | 500 | 6.23 |
| 22 | cambridge_block_7 | 970 | long | 436 | 2061.2 | 2051.2 | 155.9 | — | 0 | 500 | 13.40 |
| 23 | cambridge_block_9 | 1213 | long | 478 | 2011.9 | 2001.8 | 52.1 | — | 0 | 500 | 6.15 |
| 24 | cambridge_block_9 | 1148 | short | 51 | 2237.5 | 2227.5 | 3.9 | Y | 8 | 500 | 3.50 |

### 5.3 按长度分层

| bin | n | NE 均值 small / tm | min_dist 中位 | 进入 20 m |
|---|---:|---|---:|---:|
| short | 8 | 2180 / 2171 | 62.7 | 2/8 |
| medium | 8 | 2101 / 2094 | 93.2 | 1/8 |
| long | 8 | 2139 / 2142 | 45.7 | 0/8 |

NE 与轨迹长度**无关**（三种分层都在 2.1 km 附近），说明失败不是"步数不够"造成：
最短的一条教师只需 51 步，模型用 500 步仍然走到 2.2 km 外。

### 5.4 动作分布

| 动作 | small | transformer_m |
|---|---:|---:|
| MoveForward | 11,007 (91.7%) | 10,986 (91.6%) |
| GoDown | 993 (8.3%) | 1,014 (8.5%) |
| TurnLeft / TurnRight / GoUp / Stop | **0** | **0** |

24 × 500 = 12,000 步全部由两类动作构成。**闭环策略塌缩成"前进 + 下降"**，从不转向、
从不上升、从不停止。

### 5.5 两个模型的对比

- 逐条 min_dist **24/24 完全相同**（差 <0.05 m），水平轨迹相关系数 ≥ **0.9995**；
  差异只来自 GoDown 次数（每条差 0–17 次）造成的位姿漂移。
- 逐条 |NE 差| 均值 12.3 m，最大 85.4 m；24 条中只有 3 条动作序列完全相同。
- 汇总 NE 差 4 m（2140 vs 2136），在 24 条样本的噪声范围内，**不能声称两者有差异**。

即：在当前 token / 编码器 / 检索器设置下，**把 MEC 从 small 换成 transformer_m
没有改变闭环行为**。瓶颈不在 MEC 容量。

## 6. 失败案例：三次"路过目标"

| map | idx | min dist | 首次进入 20 m 的步 | 20 m 内停留步数 | 终点 NE | 结果 |
|---|---:|---:|---:|---:|---:|---|
| birmingham_block_12 | 817 | **7.1 m** | 40 | 13 | 2255 m | 离开后继续前进 |
| cambridge_block_18 | 1718 | **3.3 m** | 53 | 6 | 2223 m | 离开后继续前进 |
| cambridge_block_9 | 1148 | **3.9 m** | 49（tm 51） | 8 | 2238 m | 离开后继续前进 |

三条轨迹都在前 55 步内经过目标 3.3–7.1 m 处，停留 6–13 步后继续前行约 2.2 km。
**模型有能力到达目标附近，但从不输出 STOP**，这与实验 03 §3.1 的审计一致：
STOP 由 `arrival_head(current, goal)` 的 8 维匹配特征决定，不消费融合表示，
GAB / 历史 / 位姿 / 地图对它没有影响；而检索到的 goal 与当前帧的匹配度在这一步
显然没有超过联合 argmax 中最大动作的分数。

另外 6 条轨迹的 min_dist 在 20–50 m（21.1、20.4、20.9、32.3、39.3、49.4 m），
同样没有触发停止。

## 7. 结论

**已证实事实**

1. 在 500 步上限下，`val_seen` 48 条轨迹 **SR = 0**，且 48/48 全部跑满 500 步
   （`stopped = False`）。步数上限不再是解释因素。
2. 模型**从不输出 STOP**：0/48；也从不输出 Turn 或 GoUp。动作只有 MoveForward（92%）
   与 GoDown（8%）。
3. 3 条轨迹在 20 m 内停留 6–13 步后离开，最近距离 3.3 m，终点仍在 2.2 km 外。
4. NE 约为起点距离的 10 倍，且与轨迹长度分层无关。
5. `small` 与 `transformer_m` 的水平轨迹几乎重合（相关系数 ≥0.9995，min_dist 24/24
   相同），两者在当前设置下不可区分。

**未验证假设**

- 修好 arrival head（接入融合表示）后，上述 3 条 near-miss 能否转为成功——这是实验 04 的核心问题。
- 动作塌缩到 2 类在多大程度上由 Token 采样造成（实验 06 的覆盖率问题），而非 MEC 本身。
- 历史 0–7 vs 16、GAB 窗口级 vs episode 级的错配（实验 03 §3.2–3.3）对闭环指标的影响量级。
- 24 条样本的置信区间很宽（SR 的 95% 上界约 14%），不能据此估计总体成功率。

**待实验问题**

- 实验 04/05 修复后，同一批 24 条 episode 的 SR / min_dist / STOP 率如何变化。
- 是否需要把"进入 20 m 后是否离开"作为独立指标持续跟踪（本轮 3/3 离开）。

## 8. 与实验 01 / 03 的关系

- 实验 01 建议 500 步上限，理由是覆盖官方过滤后的全部 episode。本轮证实：**步数够了，
  但仍然全败**，因此 100 步截断不是 SR = 0 的唯一原因。
- 实验 03 预测"STOP 头不消费融合表示，GAB 在闭环中是死变量"。本轮从行为侧给出佐证：
  模型在 3.3 m 处不停，且动作里完全没有 STOP。

## 9. 限制

- 每划分每轮 24 条（渲染成本所限），单轮、无重复种子，置信区间宽；不做显著性声明。
- 短/中/长分层是本轮审计的三分位，不是 CityNav 官方分类。
- 单步延迟在同一台机器上随地图差异很大（0.82–13.40 s），墙钟时间不代表模型速度。
- 本报告只覆盖 `val_seen`；`val_unseen` 尚未完成，两份数据不可混用。
- 未读取 `test_unseen`。

## 10. 复现

```bash
cd experiments/02_full_closed_loop_recheck
setsid nohup bash run_experiment_02.sh > /tmp/exp02_suite_run.out 2>&1 &
# 逐条实时汇报：
setsid nohup /home/20261024_1/miniconda3/envs/DRL_migrated/bin/python -u \
  scripts/live_report.py > /tmp/exp02_live_report.out 2>&1 &
```

- 单分片命令见 `runs/val_seen_small_roundA/suite_stdout.log` 及 `closed_loop_eval.py` 的
  `--out/--journal` 参数；journal 会自动跳过已完成 episode。
- 逐条原始记录：`runs/<run>/*.jsonl`（meta + 每条 episode 一个 JSON 行）。
- 汇总：`runs/<run>/*_full_summary.json` / `*_summary.csv`。

## 11. 收尾状态

1. val_unseen round A 已完成并汇总（见 §12）；实验级结果在 `results/`。
2. 未运行 round B（BUDGET 规定每划分 1 轮）；`selection/*_roundB_indices.json` 保留未用。
3. 本实验结束后不再启动新的闭环组合；后续判断见 `../STATUS_2026-10-29.md` §6。

## 12. val_unseen round A 结果（自动汇总，2026-10-29 22:35:56）

- 完成时间：2026-10-29 22:35:56；small:full 与 transformer_m:full 各 24 条、500 步上限。
- 汇总文件：`runs/val_unseen_{small,transformer_m}_roundA/*_full_summary.{json,csv}`；
  实验级汇总：`results/summary.json`、`results/cap_sensitivity.{csv,md}`（由
  `scripts/aggregate_experiment_02.py` 生成）。

| 指标 | small:full | transformer_m:full |
|---|---:|---:|
| SR（终点 ≤20 m） | 0/24 | 0/24 |
| stopped / stopped_within_20m | 0 / 0 | 0 / 0 |
| 跑满 500 步 | 24/24 | 24/24 |
| NE 均值 / 中位 / 范围 (m) | 2165.3 / 2154.8 / 1909.1–2472.5 | 2155.1 / 2135.3 / 1909.1–2477.5 |
| 起点距离均值 (m) | 220.1 | 220.1 |
| min_dist 均值 / 中位 / 最小 (m) | 103.8 / 85.5 / 5.8 | 103.8 / 85.5 / 5.8 |
| 进入 20 m（其中离开） | 3/24（3/3） | 3/24（3/3） |
| 动作分布 | MoveForward 91.2%、GoDown 8.8% | MoveForward 90.7%、GoDown 9.3% |
| 平均单步延迟 (s) | 9.77 | 9.75 |

### 12.1 逐条结果

| # | map | episode | steps | 进入20m | NE small (m) | NE tm (m) | min dist (m) | 动作（small） |
|---:|---|---|---:|---|---:|---:|---:|---|
| 1 | birmingham_block_5 | 0783b3d1 | 500 | — | 2393.5 | 2378.4 | 147.3 | GoDown 50, MoveForward 450 |
| 2 | birmingham_block_5 | 21370c7b | 500 | — | 2232.0 | 2222.0 | 88.9 | GoDown 67, MoveForward 433 |
| 3 | birmingham_block_5 | 394f31cc | 500 | — | 2009.8 | 2009.8 | 195.5 | GoDown 61, MoveForward 439 |
| 4 | birmingham_block_5 | 4f7c8634 | 500 | — | 1977.0 | 1972.0 | 142.7 | GoDown 45, MoveForward 455 |
| 5 | birmingham_block_5 | d9b813d1 | 500 | — | 2167.1 | 2172.1 | 238.3 | GoDown 45, MoveForward 455 |
| 6 | birmingham_block_5 | e0da34d8 | 500 | — | 2243.3 | 2233.4 | 160.3 | GoDown 49, MoveForward 451 |
| 7 | cambridge_block_10 | 1a91d7c0 | 500 | Y | 2142.6 | 2122.6 | 14.8 | GoDown 36, MoveForward 464 |
| 8 | cambridge_block_10 | 6a9775a7 | 500 | — | 2186.9 | 2147.0 | 28.7 | GoDown 51, MoveForward 449 |
| 9 | cambridge_block_10 | b0640bf1 | 500 | — | 2054.6 | 2019.5 | 71.3 | GoDown 43, MoveForward 457 |
| 10 | cambridge_block_10 | b8c50cb5 | 500 | — | 2357.6 | 2347.6 | 54.1 | GoDown 33, MoveForward 467 |
| 11 | cambridge_block_10 | b8f07a31 | 500 | — | 2472.5 | 2477.5 | 66.2 | GoDown 15, MoveForward 485 |
| 12 | cambridge_block_10 | fd55c101 | 500 | — | 2408.0 | 2403.0 | 70.5 | GoDown 18, MoveForward 482 |
| 13 | cambridge_block_2 | 4648ee8a | 500 | Y | 1921.1 | 1916.1 | 12.1 | GoDown 48, MoveForward 452 |
| 14 | cambridge_block_2 | 5e291e7f | 500 | — | 2031.2 | 2021.2 | 115.3 | GoDown 54, MoveForward 446 |
| 15 | cambridge_block_2 | 80c44f4e | 500 | — | 2123.6 | 2123.6 | 82.1 | GoDown 53, MoveForward 447 |
| 16 | cambridge_block_2 | 89b2d558 | 500 | Y | 1909.1 | 1909.1 | 5.8 | GoDown 38, MoveForward 462 |
| 17 | cambridge_block_2 | 9073b657 | 500 | — | 2322.6 | 2317.6 | 60.4 | GoDown 47, MoveForward 453 |
| 18 | cambridge_block_2 | dce8c7e7 | 500 | — | 2212.6 | 2197.6 | 97.7 | GoDown 44, MoveForward 456 |
| 19 | cambridge_block_3 | 2bc06359 | 500 | — | 2105.3 | 2095.4 | 199.9 | GoDown 49, MoveForward 451 |
| 20 | cambridge_block_3 | 5c251629 | 500 | — | 2033.7 | 2023.7 | 200.2 | GoDown 51, MoveForward 449 |
| 21 | cambridge_block_3 | 5fbb8c91 | 500 | — | 2328.1 | 2313.1 | 183.0 | GoDown 44, MoveForward 456 |
| 22 | cambridge_block_3 | 6d16288f | 500 | — | 2247.6 | 2237.6 | 65.9 | GoDown 32, MoveForward 468 |
| 23 | cambridge_block_3 | c0958ce5 | 500 | — | 1973.1 | 1963.1 | 51.6 | GoDown 51, MoveForward 449 |
| 24 | cambridge_block_3 | efab1726 | 500 | — | 2114.7 | 2099.7 | 138.0 | GoDown 38, MoveForward 462 |

### 12.2 与 val_seen 的对比

| 指标 | val_seen（small / tm） | val_unseen（small / tm） |
|---|---|---|
| SR | 0/24 / 0/24 | 0/24 / 0/24 |
| STOP | 0 / 0 | 0 / 0 |
| 跑满 500 步 | 24/24 / 24/24 | 24/24 / 24/24 |
| NE 均值 (m) | 2140.0 / 2135.6 | 2165.3 / 2155.1 |
| 进入 20 m | 3/24 / 3/24 | 3/24 / 3/24 |
| min_dist 最小 (m) | 3.3 / 3.3 | 5.8 / 5.8 |

### 12.3 结论（由汇总数据直接生成）

1. val_unseen 24 条 / 模型全部跑完 500 步，**SR = 0/24（small）与 0/24（transformer_m）**；STOP 次数 0 / 0。
2. 动作仍然只有 MoveForward 与 GoDown（MoveForward 91.2%、GoDown 8.8%），Turn/GoUp/Stop 均为 0。
3. small 有 3/24、transformer_m 有 3/24 条轨迹进入过 20 m，全部离开；两模型最近距离 5.8 m。进入的轨迹：cambridge_block_2 89b2d558 (5.8 m)、cambridge_block_2 4648ee8a (12.1 m)、cambridge_block_10 1a91d7c0 (14.8 m)。
4. 与 val_seen 的每个指标同向且量级相同，未见留出划分上的行为差异；两模型在闭环中仍不可区分。

结论：500 步上限下 val_unseen 复现了 val_seen 的全部失败模式；本实验到此结束，不再加跑 round B 或新的闭环组合（见 `../STATUS_2026-10-29.md` §6）。

---

_本段由 `scripts/finalize_experiment_02.py` 于 2026-10-29 22:35:56 自动生成。_
