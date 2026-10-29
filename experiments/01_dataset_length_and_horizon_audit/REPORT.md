# 实验 01：CityNav 轨迹长度与闭环步数审计

- 日期：2026-10-29
- 状态：完成
- 数据：`original_citynav/{train_seen,val_seen,val_unseen}.json`
- 未读取：`test_unseen`
- 脚本：`scripts/audit_lengths.py`
- 结果：`results/episode_metrics.csv`、`results/summary.json`

## 1. 官方口径核查

来源：

- 论文：<https://arxiv.org/abs/2406.14240>（ICCV 2025）
- 官方代码：<https://github.com/water-cookie/citynav>
- 本地数据说明：`DATA/sbfnav-data/DATA_LAYOUT.md`

核查结论：

| 问题 | 官方事实 |
|---|---|
| 是否定义短/中/长轨迹？ | **没有**。论文只定义 Easy / Medium / Hard。 |
| Easy/Medium/Hard 定义 | 按起点到目标的距离，以训练集 33/66 分位划分：<171 m 为 Easy，171–258 m 为 Medium，>258 m 为 Hard。 |
| 是否报告平均动作数？ | **没有**。论文只报告平均路径长度：人类示范（look-ahead）508 m，最短路 290 m。 |
| 官方 episode 生成步数限制 | `vlnce/dataset/generate.py` 中 `max_steps=500`；教师动作数 >500 的 episode 不进入数据集。 |
| 官方评测步数默认值 | `vlnce/parser.py` 中 `eval_max_timestep` 默认 **200**。 |
| 成功判据 | `success_dist=20.0` m。 |
| 教师 STOP | `LookaheadTeacherParams.stop_distance_meters=5.0`，lookahead=1。 |

因此本报告不编造“官方短/中/长统计”。下文所有长度分位、教师动作数和动作分布均为本地数据实测。

## 2. 方法

- 将每条原始轨迹的 6 元组 `(x,y,z,dx,dy,dz)` 转为 5 维位姿，再用官方 lookahead teacher（lookahead=1、前进 5 m、转向 30°、升降 2 m、STOP 5 m）生成教师动作序列。
- **轨迹点数** = 原始 JSON `trajectory` 的长度；**教师动作数** = teacher 实际输出的动作条数（含终止 STOP）。两者不是一回事。
- 统计时把 teacher 的安全上限设为 1000，避免提前截断；命中 1000 的样本单独标记。
- 距离使用起点到 `target_positions[-1]` 的 3D 距离；20 m 成功判据只用于统计教师轨迹自身的成功率。
- 并行 16 进程，27,326 条 episode 总计约 26 秒。

## 3. 各划分统计

| split | n | 轨迹点数 mean/med/p95/max | 教师动作数 mean/med/p95/p99/max | 人类路径 m mean | 教师路径 m mean | >100 | >200 | >300 | >400 | >500 |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| train_seen | 22,002 | 88/63/249/1220 | 197/159/435/650/1001 | 551 | 545 | 79.9% | 37.0% | 16.9% | 6.8% | 2.9% |
| val_seen | 2,498 | 88/63/235/900 | 196/160/431/609/1001 | 546 | 542 | 79.3% | 38.6% | 16.7% | 7.0% | 2.6% |
| val_unseen | 2,826 | 70/52/188/1112 | 186/159/377/543/1001 | 510 | 519 | 83.3% | 34.3% | 11.5% | 3.7% | 1.3% |

补充：

- 轨迹点数与教师动作数的 Pearson 相关仅 **0.19（val_seen）/ 0.27（val_unseen）**；点数中位数 63、动作数中位数 160，约 **2.5–3 倍**关系。把轨迹点数当动作数会严重低估所需步数。
- 平均人类路径长度 546 m（val_seen）/ 510 m（val_unseen），与论文的 508 m 同量级，说明本地数据的长度口径与官方一致。
- 教师以 5 m 规则 STOP 后，按 20 m 判据的成功率约 **89.7%**；约 10.3% 的轨迹在 5 m STOP 时距 `target_positions[-1]` 仍 >20 m，与实验 04 要处理的 STOP 语义冲突一致。
- 命中 1000 步安全上限的样本：train 26 条、val_seen 3 条、val_unseen 2 条；这些属于超长轨迹，官方 `max_steps=500` 本来就会把它们过滤掉。

## 4. 按官方难度分层

| split | difficulty | n | 教师动作 mean/med/p95 | >100 | >200 | >400 |
|---|---|---:|---|---:|---:|---:|
| val_seen | easy | 836 | 149/108/376 | 53.7% | 22.2% | 4.5% |
| val_seen | medium | 807 | 198/162/417 | 84.4% | 38.0% | 6.1% |
| val_seen | hard | 855 | 241/215/453 | 99.6% | 55.0% | 10.2% |
| val_unseen | easy | 773 | 143/113/314 | 56.8% | 21.1% | 2.1% |
| val_unseen | medium | 972 | 174/147/341 | 86.1% | 28.2% | 2.2% |
| val_unseen | hard | 1,081 | 227/198/423 | 99.7% | 49.2% | 6.2% |

- 即便 Easy 也有超过一半 episode 需要 >100 步；Hard 几乎全部 >100 步。
- 当前闭环的 **100 步上限只覆盖约 20% 的 episode**，会系统性截断 Medium/Hard。

## 5. 动作分布（教师动作）

| split | Stop | MoveForward | TurnRight | TurnLeft | GoUp | GoDown |
|---|---:|---:|---:|---:|---:|---:|
| train_seen | 0.5% | 32.6% | 5.1% | 5.2% | 16.2% | 40.4% |
| val_seen | 0.5% | 32.5% | 5.2% | 5.1% | 16.1% | 40.5% |
| val_unseen | 0.5% | 33.3% | 4.7% | 4.8% | 15.2% | 41.5% |

这与论文 Figure 4(f) 的动作直方图方向一致（GoDown 最多、Stop 极少），也说明后续训练中的动作不平衡是数据本身的性质，不是本轮新引入的。

## 6. 步数上限结论

**当前 100 步上限不合理。**

- 原始 episode 中 80.2% 的教师动作数 >100；val_seen 79.3%、val_unseen 83.3%。
- 官方 episode 生成在 500 步过滤，说明官方认为 500 步以内才是有效 episode；官方评测默认 200 步，但这不是数据分布决定的硬上限。
- 400 步仍会截断约 6.5% 的原始 episode（val_seen 7.0%、val_unseen 3.7%）；500 步覆盖官方过滤后的全部 episode。

**固定全局上限建议：500 步。** 理由：

1. 与官方 episode 生成的 `max_steps=500` 一致；
2. 覆盖官方过滤后的全部 episode，不针对单条测试轨迹动态设限；
3. 100/200/300/400 作为敏感性对照，用于量化“短上限”造成的损失。

该上限不依赖测试轨迹的真实长度，也不向 MEC 提供任何真值信息；只是评测器统一的步数预算。

## 7. 对实验 02 的含义

- 必须把 100 步和更长上限分开评估，否则无法判断 SR=0 是策略问题还是截断问题。
- 实验 02 将使用 500 步固定上限，并在 100/200/300/400/500 处分别计算 SR、NE、是否进入 20 m、是否离开 20 m、是否在目标附近 Stop。
- 本轮仍只测 `small:full` 与 `transformer_m:full`，不测 `map_free`。

## 8. 限制

- 本地 `original_citynav` 是原始 MTurk 轨迹，不是官方 `generate.py` 过滤后的最终 episode 集；报告同时给出原始和 ≤500 两种口径。
- 教师动作由本地 `TrajectoryAdapter` 复现官方 lookahead teacher；已在代码注释中与官方 `LookaheadTeacherParams(lookahead=1)` 对齐，但不排除插值细节差异。
- 未读取 `test_unseen`。
