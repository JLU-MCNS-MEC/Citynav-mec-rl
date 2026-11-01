# Stage 11：纯教师监督多地标几何消融（M-v4 / M-v5）— 状态与协议

- 分支：`exp/citynav-mec-fixed50m-topdown`
- 训练池：11A 已完成 → **18,521 条 episode / 1,279,882 个决策状态**（纯 Stage 02 High+Medium ∩ Stage 03 成功）
- 只训练 seed42。

## 数据来源红线（2.x）

- 训练样本仅来自：原始 CityNav 人工轨迹 → Stage 02 质量审计 → Stage 03 四动作教师 → 教师成功回放 → 视觉及目标几何特征 → 监督训练。
- **禁止**：03B 直接目标导向教师轨迹、10C 的 50k C1 控制器闭环状态及动作、任何 Val/Test 教师动作标签、真实目标坐标/距离/未来位姿/成功标志作为模型输入。
- C1 仅作为冻结特征模块（Layer2 描述子、实时匹配、09C-1 信念、位置不确定性）。
- 不引入 09C-2 硬门控，不重测 C2。
- M-v3 的 250k 混合样本与权重完整保留作历史对照，不复用为训练数据。

## 各阶段状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| M0 | 服务器检查、数据恢复与再生 | ✅ 完成（官方原始数据哈希校验通过；four_action_labels/candidates24/candidates48 确定性再生并验证） |
| 11A | 纯教师数据集审计 | ✅ 完成（18,521 条 / 1,279,882 状态，已推送 `913b6723`） |
| 11B | M-v4 数据重建 + 训练 | ⏳ 进行中（特征重建 + 物化 + 训练） |
| 11C | M-v5 数据物化 + 训练 | 待 11B 完成 |
| 11D | 闭环评测 + 报告 + 推送 | 待 11C 完成 |

## ⚠️ 关键发现（2026-11-01）：10C/10D 特征管线建立在 03B 轨迹上

`dataset_v2.load_split` 明确读取 03B 直接目标导向教师 npz（"Read the 03B
direct-teacher trajectories"）；实测 10C 缓存轨迹长度与 Stage 03 四动作教师
严重不符（如 16 行 vs 201 步），转弯占比 ~3%（四动作教师 ~12%）。因此：

- **10C 缓存（79 GB）不能用于 Stage 11 训练特征**（03B 轨迹是文档 2.1 明令
  禁止的数据），M-v4/M-v5 的特征必须从 `four_action_labels.npz`（Stage 03
  四动作教师）重新因果回放（`build_features_11b.py`，C1/09C 计算与 10C 完全
  相同，仅轨迹来源不同）。
- 训练数据纯教师监督、无任何混合（对应"不要和 10D 一样用混合数据集训练"）。
- Val 教师（仅验证用）：`build_val_teacher_11b.py` 用 Stage 03 同一转换代码
  在 val_seen 上构建四动作教师（不进入训练集）。
- 10C 缓存与 M-v3 权重/报告作为历史对照完整保留。

## 11B M-v4 协议（部分为推断，标注 ⚠️ 假设）

- **数据**：11A 全量池（18,521 集 / 1,279,882 状态，无每集上限、无重采样）；特征由 `build_features_11b.py` 从 Stage 03 四动作教师因果回放重建（纯教师监督，无 03B / 无 C1 闭环状态 / 无混合）；物化为 memmap `.npy` + 分块 DataLoader（文档 4.3 强制）。
- **模型**：Transformer-M（10D 架构 d_model=256 / 4 层 / 8 头），三处修复（用户已批准 2026-11-01）：
  1. 历史时间位置编码：为 16 个历史槽位增加可学习时间嵌入（v3 缺失；已实现 `hist_time`）
  2. 重复特征：belief[8:10]==match[0:2]（dx/dy）、belief[4]==match[2]（score）重复 → 融合状态去重为 12 维（已实现 `BELIEF_IDX`/`MATCH_IDX`）
  3. 训练调度：OneCycleLR 改为按 batch 步进（total_steps=epochs×steps_per_epoch，标准做法）+ NaN 防护（已实现）
- **超参**（沿用 10D）：batch 128、epochs 30 + patience 10、seed 42。
- **输入**：与 M-v3 完全一致（当前 7×7×2048 空间视觉 + 16 步历史 + C1 目标描述子 + 09C-1 信念 + 匹配测量 + 位姿/动作历史）；**不含多地标几何**。

## 11C M-v5 协议（⚠️ 全部推断，待用户确认）

- 数据与超参与 M-v4 完全一致，仅增加**多地标几何特征**：取候选库中 top-k 地标的相对距离、相对方位及地标间空间关系（pairwise 距离/夹角），作为额外 token 输入 ⚠️ k 与编码形式待定。

## 11D 闭环评测（⚠️ 推断，沿用 10E 机制）

- Val-Seen / Val-Unseen / Test-Unseen 三 split 闭环，M-v4 / M-v5 各跑一遍；指标 SR 为主，对比 C1（64.7%）、10E M-v3（55.4%）；产出对比表、逐地图分析、成功/失败视频（沿用 10E 脚本与保存约定）。

## 保存与推送约定（沿用之前指令）

- 各阶段结果：`experiments/11_teacher_geometry_ablation/<阶段>/results/` + 中文 `REPORT.md` + `figures/`。
- 大产物（memmap、checkpoints、视频）gitignore，不入库；代码、报告、清单入库。
- 每个阶段完成后 commit + push（身份 `ty-zhang-jlu`，HTTPS 走 mihomo 代理）。
