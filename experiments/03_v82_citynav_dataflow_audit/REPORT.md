# 实验 03：MEC-V8.2 与 CityNav 数据流逐项对照

- 日期：2026-10-29
- 状态：**完成**（代码审计 + 伪 STOP 全量统计）
- 依赖：实验 01；实验 02 的闭环结果另行汇总
- 未读取：`test_unseen`

## 1. 假设与目的

在调整任何结构或训练协议之前，先确认两件事：

1. CityNav-MEC 的每个输入在**训练**与**闭环**中是否同源、同维度、同更新时机；
2. STOP 标签与 STOP 决策到底由什么产生，是否存在 `max_steps` 触顶产生的伪 STOP，
   以及 STOP 头是否消费了融合表示。

审计只读，不修改代码；全部结论标注为「已证实事实」或「未验证假设」。

## 2. 伪 STOP 标签（全量统计）

教师默认 `TeacherConfig.max_steps=400`（`src/config.py:230`，
`configs/train_config.example.yaml:36` 同为 400）。教师循环达到上限时**直接追加
STOP**（`src/data/trajectory_adapter.py:354,423`，`stop_reason="max_steps"`），
不检查是否真的到达目标（教师 STOP 距离为 5 m，成功半径为 20 m）。

### 全量（27,326 条，16 线程审计，结果在 `results/full_max400|500/`）

| cap | split | episode | 触顶 | 触顶且终点 >20 m（伪 STOP） | 触顶终点距离 均值/中位/p95/max (m) |
|---|---|---:|---:|---:|---|
| 400 | train_seen | 22,002 | 1,505 (6.84%) | **1,374 (6.24%)** | 115.4 / 89.5 / 303.9 / 536.5 |
| 400 | val_seen | 2,498 | 174 (6.97%) | 151 (6.04%) | 107.9 / 83.1 / 302.7 / 433.2 |
| 400 | val_unseen | 2,826 | 104 (3.68%) | 95 (3.36%) | 116.3 / 89.9 / 303.9 / 445.9 |
| 500 | train_seen | 22,002 | 642 (2.92%) | 595 (2.70%) | 125.3 / 97.8 / 320.6 / 451.1 |
| 500 | val_seen | 2,498 | 66 (2.64%) | 58 (2.32%) | 110.2 / 85.1 / 251.5 / 317.5 |
| 500 | val_unseen | 2,826 | 38 (1.34%) | 35 (1.24%) | 134.5 / 121.1 / 285.1 / 384.1 |

- 每条 episode 恰好一条 STOP 监督，因此上表比例即「STOP 标签中的伪 STOP 比例」：
  cap 400 时 train_seen **6.8%**，cap 500 时 **2.9%**。
- 这些标签的终点中位距离 **89.5 m**，远超 20 m 成功半径——模型被训练成在离目标
  很远处停止。

### 对现有 token 缓存的实际污染

`data_cache/t5_frozen_1000/tokens/t5_tokens.json` 共 12,800 帧、800 条 STOP 监督
（train 625 / val_seen 35 / val_unseen 140，每条 episode 一条）。以 STOP 帧
`teacher_step == 400` 判定触顶产生：

| split | STOP 帧 | 来自伪 STOP | 占该 split STOP 监督 |
|---|---:|---:|---:|
| train_seen | 625 | 2 | 0.32% |
| val_seen | 35 | 0 | 0% |
| val_unseen | 140 | 6 | **4.3%** |
| 合计 | 800 | 8 | **1.0%** |

污染比例远低于全量，原因是训练缓存只覆盖 **8/24 张地图**，且这些地图轨迹偏短
（缓存内 episode 最大 teacher_step 中位数仅 112）。这与实验 06 的地图覆盖率问题
是同一个根因，**不能**据此认为伪 STOP 已解决。

### 闭环不会制造伪 STOP

`closed_loop_eval.py:178-220` 只在模型 argmax 为 STOP 时终止；达到 `--max-steps`
只记 `max_steps_reached=True`、`stopped=False`，不追加 STOP 标签，也不强制 STOP。
**已证实事实。**

## 3. CityNav 侧代码审计（训练 vs 闭环）

### 3.1 STOP 头绕过融合表示（高风险，已证实）

- full 模型：`current` 与 goal/history/gab/map/lang 拼成 `[B,584,256]` 过
  `fusion_encoder`，`cls_feat` 只送 `move_head`（`src/models/t5_mec.py:494-497`）；
  arrival 走 `self.arrival_head(current, goal_tokens.float())`
  （`t5_mec.py:499-501`）——输入是**融合前的原始 token**。
- small:full 同样：`arrival_head(current_tokens, goal_tokens)`（`t5_mec.py:315-318`）。
- `MatchArrivalHead`（`t5_mec.py:86-88`）→ `match_features`
  （`t5_mec.py:44-72`，相似度来自 `src/models/mec_common.py:27-36`）→ 8 维手工特征
  （max/top10/mean/entropy/weighted_xy/best_xy）→ MLP 8→64→32→1。

**含义：GAB、历史、位姿、地图特征对 STOP 概率没有任何影响；能改变 STOP 的只有
当前帧视觉 token 与 goal token 的匹配度。** 这与实验 04 要修的 STOP 语义直接相关。

### 3.2 历史长度的训练/闭环错配（高风险，已证实）

- 训练：`--history-context` 默认 **0**（`scripts/t5_smoke.py:177`），窗口为 8 步
  （缓存实际参数），因此每条样本的有效历史长度只有 **0..7**。
- 闭环：`history_arrays(..., steps=16)`（`closed_loop_eval.py:146-150`；
  `src/data/t5_features.py:150-152`），历史随 episode 增长直到 16 槽。
- 结果：闭环第 8–15 槽是训练中从未出现的输入，属于分布外。

### 3.3 GAB 的更新时机错配（高风险，已证实）

- 训练：GAB 在每个窗口内重置（`t5_smoke.py:537-543`；代码注释明确写了
  "GAB stays window-local"，`src/data/gab.py:198`），同一帧最多 7 次证据更新。
- 闭环：`GABState` 从 episode 开始持续累积（`closed_loop_eval.py:130,164`；
  `src/data/gab.py:117-154`），同一帧可能经过数百次更新。
- 阈值本身一致（从 token 元数据读取，`closed_loop_eval.py:358-368`）。

### 3.4 决策规则（中风险，已证实）

闭环用 `logits.argmax`（`closed_loop_eval.py:172-174`），而 logits 来自
`combine_move_and_arrival`（`mec_common.py:159-169`）的对数概率拼接，等价于
「STOP 概率 vs 最大动作概率」，**不是** `arrival_stop_threshold=0.5`
（`src/config.py:305`）的单阈值规则。训练评估用同一 joint argmax
（`src/training/supervised.py:105-106`），两者自洽，但报告中不能把它描述成
"STOP 概率 > 0.5"。

### 3.5 其它输入

| 输入 | 训练 | 闭环 | 风险 |
|---|---|---|---|
| visual token | teacher 位姿渲染，float16 存盘（`t5_smoke.py:477-491,576`） | 执行位姿渲染，float32（`closed_loop_eval.py:137-143`） | 低（token 对齐实测 cos=1.0000） |
| goal / top5 | retriever 检索，整段 episode 固定 | 同（`closed_loop_eval.py:110-118`） | 低 |
| map_feat | `map_features_from_scores`，相对坐标 | 同（`closed_loop_eval.py:165-167`） | 低 |
| previous action | teacher 动作 | 模型实际动作（off-policy） | 中 |
| pitch | 每步替换为最近示范点 pitch（`trajectory_adapter.py:406-415`） | 保留初始 pitch（`closed_loop_eval.py:154-197`） | 中（量级未验证） |

### 3.6 因果与泄漏

闭环输入只有 current/history/language/goal/gab/map_feat（`closed_loop_eval.py:151-170`）；
`target_xyz` 只用于距离与指标（`:125-129,186-213`），不进入模型。
**没有**发现目标或未来轨迹泄漏。

唯一隐患：retriever 的最优 checkpoint 按 val_seen + val_unseen 的 Recall@1 选择
（`t5_smoke.py:1415-1425`；`src/training/retrieval.py:299-315`），评估划分参与了
模型选择。target 仅作检索标签，未进入闭环输入；是否虚高**未验证**。

## 4. MEC-V8.2 侧对照

| 维度 | MEC-V8.2 | CityNav-MEC |
|---|---|---|
| 时间建模 | 单帧前馈，无历史（`gbs_model_v3.py:10-16`） | 16 步历史 + GAB |
| 输入模态 | current token、goal token、3×8×8 heatmap、5 维 network | token、goal、GAB、map、pose history、language |
| STOP 头 | 104 维手工匹配特征 + MLP，不消费 fused（`gbs_model_v3.py:545-552,109-116`） | 8 维匹配特征 MLP，同样不消费 fused（`t5_mec.py:499-501`） |
| 伪 STOP | **不存在**：超时轨迹整条丢弃（`token_data.py:300-311,786-789`） | 存在：cache 内 1.0%，全量标注 6.24% |
| 标签分布处理 | 远端负样本 >60 m 仅保留 3%（`config.py:141-142`）；约 6.8% 瞬移合成硬状态（`token_data.py:334-359`） | 无降采样；采样改变动作比例（实验 06） |
| 泄漏面 | encoder 预训练用 oracle goal mask（`train_shared_encoder.py:157`） | 无目标泄漏；retriever 用 val 选模 |

**结论：两侧在 STOP 上的失败机制相似（都靠单帧匹配度），但 V8.2 没有历史，也没有
伪 STOP。因此 CityNav 的闭环结果不能和 V8.2 的数字直接对标。**

## 5. 结论

**已证实事实**

1. 教师伪 STOP 存在且规模明确：cap 400 下 train_seen 6.24% 的 episode 在终点
   >20 m 时被标成 STOP；cap 500 降到 2.70%。
2. 现有缓存中伪 STOP 只占 STOP 监督的 1.0%，但这是因为缓存只覆盖 8 张短轨迹地图，
   不代表问题已解决。
3. full 与 small 的 STOP 头都不消费融合表示；GAB/历史/位姿对 STOP 无影响。
4. 训练历史有效长度 0–7，闭环到 16；GAB 训练窗口内重置、闭环节点累积——两处都是
   分布错配。
5. 闭环不制造伪 STOP，也没有目标泄漏。
6. V8.2 无伪 STOP、无历史，STOP 头同为侧分支，但输入构成不同，不可直接对比。

**未验证假设**

- 历史/GAB 错配对闭环指标的影响量级（需要实验 04 统一协议后对比）。
- pitch 处理差异对闭环成功率的影响。
- retriever 用 val 选模造成的指标虚高幅度。

**待实验问题**

- 把 fused 表示接入 arrival head 后，STOP 的召回与误停如何变化（实验 04/05）。
- 伪 STOP 按 5 m/20 m 重新标注后，STOP 监督的类别比例如何变化。

## 6. 对后续实验的影响

1. **实验 04 的 STOP 修复**：不能只调阈值；需要把 arrival head 接到融合表示
   （或至少接入 GAB/map），否则 GAB 在闭环中对 STOP 永远是死变量。
2. **实验 04 必须同时统一历史与 GAB 协议**：训练侧把 `history_context` 提到 16、
   GAB 改为 episode 级累积（或闭环改为窗口内重置），否则闭环只能解释为分布外。
3. **实验 05 的过拟合诊断**：在 3.1–3.3 修好之前，Transformer-M 的过拟合诊断会被
   协议错配混淆，结论不可用。
4. **实验 06（OPEN/BLOCKED）** 不变：伪 STOP 只占缓存 1%，优先级低于采样覆盖率问题，
   但采样方案必须同时处理伪 STOP 标签。

## 7. 复现

- 伪 STOP 全量统计：`results/full_max400/`、`results/full_max500/`
  （`episode_metrics.csv` + `summary.json`），由
  `experiments/01_dataset_length_and_horizon_audit/scripts/audit_lengths.py`
  以 `--workers 16 --max-steps 400|500` 生成。
- 缓存帧统计：对 `data_cache/t5_frozen_1000/tokens/t5_tokens.json` 的
  `frames_metadata` 按 `split`、`action==0`、`teacher_step` 聚合。
- 代码审计：只读，未修改任何文件；引用行号基于 commit `745c9b0`。
