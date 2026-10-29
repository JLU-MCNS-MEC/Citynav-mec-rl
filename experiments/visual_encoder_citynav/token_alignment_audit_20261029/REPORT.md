# Token 数据与监督目标一致性审计（2026-10-29）

本审计只读共享原始数据，所有新缓存/日志/结果都在
`experiments/visual_encoder_citynav/token_alignment_audit_20261029/` 下。
现有实验（`sqrt` 闭环、MEC 权重、冻结编码器）未停止、未覆盖。

## 0. 结论摘要

| 问题 | 结论 | 证据 |
|---|---|---|
| 1. 渲染差异是否显著？ | **是，显著**；闭环与 Token 完全一致，差异在“预训练 vs Token” | RGB 平均差 19.3/255、35.1% 像素差 >10、token cos 0.928；配置见 §2 |
| 2. 多少教师 STOP 与 20 m 判据冲突？ | Train-Seen **9.79%**（2,007 / 20,497）；Val-Seen 10.67%；Val-Unseen 11.46% | §3 |
| 3. 历史上下文不一致是否影响训练？ | 现有训练历史仅 0–7 步（均值 ~3.5），闭环最多 16 步；修正后因果性 6/6 测试通过，但 **VU 准确率没有提升** | §4–§5 |
| 4. 修正 Token 后 Transformer-M 是否改善？ | **没有**。h16 的 VU acc 0.484±0.007，低于 h0 的 0.517±0.040；pretrain 渲染更低（0.462±0.018） | §5 |
| 5. 是否需要重训 E2-FOV-Pose？ | **本次证据不支持**。统一预训练渲染后指标反而下降；主要失败是 Stop/转向坍缩，不是 token 分布 | §2、§5–§6 |

## 1. 资源与隔离

- 审计开始时 GPU 由现有闭环分片占用（8 进程，约 35 GB 显存），本审计只用小批量
  GPU 编码与 CPU 渲染，未争用其显存配额。
- 所有 Token 子集、重渲染缓存和模型权重都在本目录 `cache/` 下，并被本目录
  `.gitignore` 排除；共享 `data_cache/` 与 `runs/` 未被修改。
- 新模型只写入 `cache/**/mec_audit_s<seed>/`，不覆盖现有 MEC。

## 2. A. 渲染一致性

### 2.1 三套配置（代码级对照）

| 项 | E2-FOV-Pose 预训练 | Token 生成 (`t5_smoke.py`) | 闭环 (`closed_loop_eval.py`) |
|---|---|---|---|
| 图像尺寸 | 336 | 336 | 336 |
| 点云来源 | `PointCloudStore`，**每图最多 4,000,000 点**（等步长下采样） | 原始 PLY，120 m 水平裁剪（约 13.8M 点） | 同 Token |
| 裁剪半径 | 默认 **300 m**（整图点云） | **120 m** | 120 m |
| `splat_radius` | **1**（`e2_fov_pose.yaml`） | **0** | 0 |
| near / far | 0.5 / 300 m | 0.5 / 300 m | 0.5 / 300 m |
| HFOV / pitch | 90° / trajectory | 90° / trajectory | 90° / trajectory |

### 2.2 同位姿对照渲染（`cambridge_block_25`，4 个教师状态）

| 配置 | 有效像素比 | 点数 | RGB 平均差 | >10 像素占比 | token 余弦 |
|---|---:|---:|---:|---:|---:|
| 预训练（4M/300 m/splat 1） | 0.547 | 3.85 M（整图） | 19.30 | 35.1% | 0.928 |
| Token（原始/120 m/splat 0） | 0.493 | 13.79 M | 0 | 0 | 1.000 |
| 闭环（同 Token） | 0.493 | 13.79 M | 0 | 0 | **1.0000** |
| Token 配置 + 4M 点云 | 0.432 | 2.76 M | 10.75 | 19.2% | 0.958 |

- 逐状态 token 余弦范围 **0.787–0.995**；最差状态深度平均差 1.67 m。
- 点云密度单独贡献约一半差异（0.958 vs 0.928）；`splat=1` 与 300 m 视野贡献其余。
- 图：`figures/render_pose*.png`；原始表：`results/render_consistency_cambridge_block_25.csv/json`。

**含义**：冻结编码器在“稀疏 + 加宽点 + 300 m”图像上训练，却在“稠密 + 无 splat +
120 m”图像上被推理。这是分布偏移，不是 Token 与闭环之间的不一致。

## 3. B. 教师 STOP 标签 vs 20 m 成功判据

教师 STOP 触发条件是到 **marker** 的 3-D 距离 < 5 m（`TeacherConfig.stop_distance_m`），
官方成功率用最终位姿到 **`target_positions[-1]`** 的 3-D 距离 < 20 m
（`success_radius_m`）。两者不是同一个点。

| split | 记录数 | 教师 STOP | 距目标中位 | p90 | p95 | >20 m | 冲突率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train-Seen | 22,002 | 20,497 | 6.54 m | 19.80 m | 24.51 m | 2,007 | **9.79%** |
| Val-Seen | 2,498 | 2,324 | 见 JSON | — | — | 248 | **10.67%** |
| Val-Unseen | 2,826 | 2,722 | 见 JSON | — | — | 312 | **11.46%** |

- 最大冲突距离 34.3 m（Train-Seen）。
- 另有 1,505 条 Train-Seen 轨迹以 `max_steps` 结束（6.84%），不是 5 m 规则触发。
- 候选离线规则“仅保留距目标 ≤20 m 的 STOP 标签”会移除 9.79% 的 STOP 监督，
  但**本审计不修改任何原始标签或官方评估标准**。
- 图：`figures/teacher_stop_distance.png`；全量表：`results/teacher_stop_<split>.csv/json`。

## 4. C. 历史上下文与数据覆盖

### 4.1 现有构造

- 每条轨迹取 2 个 8 步窗口（`total//3` 与 `total-8`）；`history_arrays` 只能看到
  窗口内部：每个监督步的有效历史长度 = 窗口内位置（0..7）。
- 实测 12,800 帧：历史长度分布严格为 `{0:1600, 1:1600, …, 7:1600}`，均值 3.5，
  **最大 7**；而模型输入预留 16 槽，闭环最多提供 16 步。
- 训练覆盖率：625 / 22,002 条 Train-Seen（2.84%），8 / 24 张地图；
  全量中位轨迹 **159 步**，但每条只监督 16 步（约 **10%**）。
- **采样偏置（全量 vs 采样）**：

| 动作 | 全量 Train-Seen（4.16M 动作） | 采样 10,000 帧 | 偏差 |
|---|---:|---:|---|
| Stop | 0.53% | 6.25% | **12× 过采样** |
| GoDown | 40.27% | 50.13% | 1.24× |
| MoveForward | 32.77% | 29.65% | 0.90× |
| GoUp | **16.22%** | **3.72%** | **0.23× 欠采样** |
| TurnLeft | 5.11% | 5.32% | ≈1 |
| TurnRight | 5.10% | 4.93% | ≈1 |

- 原因：两个固定窗口（`total//3`、`total-8`）系统性地漏掉了轨迹早段。
  1,200 条教师轨迹中，GoUp 的平均相对位置 0.321（45.6% 发生在前 20%），
  GoDown 为 0.586，Stop 恒在 0.994。因此“早段上升”几乎不被采样，
  而“末段下降 + 终止”被反复采样。

### 4.2 修正构造

保留同样的 8 个监督步与标签，为每个监督步渲染其之前最多 16 个真实状态，
用实际教师动作构建严格因果历史。实现为 `t5_smoke.py --history-context N`
（默认 0，行为不变），审计缓存在独立目录。

### 4.3 因果/防泄漏测试（`scripts/test_history_leakage.py`）

6/6 通过：
- 历史槽位与 `index-1-k` 的动作一一对应；
- 修改当前/未来动作不影响历史；
- 修改未来位姿不影响历史；
- 当前位姿只改变向量参考系与“进入当前状态”的最近一跳；
- 视觉历史只含过去状态；掩码长度恰为 `min(index, 16)`。

## 5. 小规模 Transformer-M 对照实验

### 5.1 设计

同一批 80 条训练轨迹 + 12 / 12 条验证轨迹（每条 2 个 8 步窗口）：

| 项 | 值 |
|---|---|
| 监督帧 | 1,280 train / 192 val-seen / 192 val-unseen |
| 动作分布 | GoDown 52.3%、Forward 27.9%、Stop 6.25%、TurnLeft 5.3%、TurnRight 4.6%、GoUp 3.7% |
| 对照 arm | h0-token（现有构造）、h16-token（修正历史）、h0-pretrain（统一预训练渲染） |
| 模型 | Transformer-M full（检索 + 地图） |
| 训练 | 1000 步、batch 16、lr 1e-4；seed 42/43/44 |
| 评估 | 与训练相同的 192 帧 val-seen / 192 帧 val-unseen，逐类召回 |

所有 arm 的标签、poses、goal/map/language 输入完全相同，只改变 Token/历史/渲染。

### 5.2 结果

三个 arm 各 3 个 seed（42/43/44）已完成，机器可读结果在
`results/model_arms.csv`，逐 seed 明细在 `results/train_*.json`。

| arm | n seeds | VU acc | VU move | VU Stop P/R | TurnRight | TurnLeft | GoUp | GoDown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h0_token | 3 | 0.517±0.040 | 0.552±0.042 | 0.000±0.000 / 0.000±0.000 | 0.067±0.094 | 0.000±0.000 | 0.083±0.118 | 0.844±0.091 |
| h16_token | 3 | 0.484±0.007 | 0.517±0.008 | 0.000±0.000 / 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.083±0.118 | 0.837±0.029 |
| h0_pretrain | 3 | 0.462±0.018 | 0.493±0.019 | 0.000±0.000 / 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.769±0.059 |

Val-Seen 均值：h0_token 0.571、h16_token 0.585、h0_pretrain 0.535；
Val-Unseen 上三者分别为 0.517 / 0.484 / 0.462。

- **修正历史上下文没有带来泛化收益**：h16 的历史长度从均值 3.5、最大 7 修正为
  全部 16，Val-Seen 略升（0.571→0.585），但 Val-Unseen 反而下降
  （0.517→0.484）。差异小于 h0 自身的 seed 方差（±0.040），不能称为改善。
- **统一预训练渲染反而更差**：h0_pretrain 的 VU acc 0.462、move acc 0.493，
  GoDown 召回 0.844→0.769，GoUp 召回降为 0。若渲染分布偏移是主因，
  这个 arm 应该改善，但结果方向相反。
- **主导失败模式完全不变**：9 个训练 run 的 Stop 召回全部为 0，TurnLeft 全为 0，
  TurnRight 只在 h0_token 的一个 seed 上非零（0.20）；所有 arm 都由
  GoDown/MoveForward 主导。

## 6. 判断：是否需要重新训练 E2-FOV-Pose

**本次审计的证据不支持“因为 token/渲染不一致而重训编码器”。**

- 渲染差异确实显著（§2），但把 Token 改成预训练协议后，三个 seed 的 VU 准确率
  反而从 0.517 降到 0.462，说明冻结编码器对这套渲染差异并不敏感；
- 历史上下文修正（§4）也没有改善 VU；因果性测试只证明构造正确，不证明收益；
- 三个 arm 的 Stop/TurnLeft 召回都恒为 0，说明主要瓶颈是**监督信号与目标函数**
  （Stop 与转向样本在窗口中稀少且被 GoDown/MoveForward 淹没），而不是 token 的
  像素分布。

因此下一步优先做定向的监督修正（Stop/转向的过采样、窗口采样修正、focal/代价敏感
损失或更长训练），而不是重渲染或重训 E2-FOV-Pose。该判断受小规模限制：80 条训练
轨迹、1280 帧、1000 步、单一 `transformer_m:full`，不能替代全量实验；但它足以说明
“渲染不一致”不是当前最值得投入的方向。

## 7. 复现

```bash
# 渲染一致性
python scripts/audit_render_consistency.py --map cambridge_block_25 --poses 4
# 教师 STOP
python scripts/audit_teacher_stop.py --split train_seen
# 历史子集（H=0/16）
H=16 SHARDS=5 bash scripts/build_history_subset.sh
# 因果测试
python scripts/test_history_leakage.py
# 预训练协议重渲染 + 合并 + 训练
python scripts/retokenize_pretrain.py --shard-dir ... --token-meta ... --out-dir ...
python scripts/merge_retokenized.py --base-cache ... --parts-dir ... --out-cache ...
python scripts/train_from_tokens.py --cache-dir ... --out ...
```
