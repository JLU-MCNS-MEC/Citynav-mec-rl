# T5 + 冻结 BERT 地图候选检索 Smoke 报告

- 日期：2026-10-27
- 数据：`train_seen` / `val_seen` / `val_unseen`；`test_unseen` 未读取
- 性质：小规模检索、防泄漏、六模态前向和单步训练/短训练 Smoke
- **不是正式训练结果，不构成地图辅助增益结论**

## 1. 输入模态（与 ToyEnv V8.2 T5 对齐）

| 模态 | 维度 | 说明 |
|---|---|---|
| Current RGB Tokens | 576 × 384 | 第一视角 RGB 经冻结视觉编码器 |
| Goal RGB Tokens | 3 × 384 | 语言检索的 Top-3 匿名地图候选视觉 Token |
| History Visual | 16 × 768 | 最近 16 步 mean+max 视觉描述符 |
| Pose History | 16 × 12 | 6 连续相对运动/相对位姿 + 6 动作 one-hot |
| Action History | 16 × 6 | 已执行六动作 one-hot |
| History Mask | 16 | 因果历史 mask |
| GAB | 20 | 8-bin belief + last reliable 8 + confidence/entropy/age/valid |
| Top-5 Map Retrieval | 22 | 5 × (相似度, 前向, 侧向, 距离) + Top-1/margin |
| Language Features | 768 | 冻结 `bert-base-uncased` |

不输入 5D Network State；Depth 仅用于渲染诊断，不进入 MEC。

## 2. 无泄漏候选库与检索

候选库只由**官方正射图 + 固定网格 + 固定尺度**生成，不读取 CityRefer 目标框、目标 ID、`object_ids`、`ann_ids` 或 `target_positions`：

- 12 个开发地图；
- 1,602 个候选；
- stride 48 m；
- side length 32 / 64 / 128 m；
- 每候选 96×96 正射切块，视觉编码器 mean token 384-D；
- 候选中心位置仅用于生成相对前向/侧向/距离特征。

检索器：冻结 BERT 768 → 轻量投影到 384，与候选视觉 token 做余弦匹配。视觉编码器从 MEC-V8.2 `shared_encoder.pt` 初始化，检索阶段允许训练，随后冻结。训练标签仅由 `train_seen` 的 `target_positions` 生成（候选中心距目标 ≤40 m 为正）；推理函数 `retrieve_candidates(language_features, candidate_embeddings, retriever, top_k)` **没有 target/ID/crop/teacher/success 参数**。

| split | samples | with positive | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|---:|
| val_seen | 23 | 23 | **0.348** | **0.652** | **0.739** |
| val_unseen | 48 | 48 | **0.021** | **0.146** | **0.292** |

（训练：128 条 train_seen episode，80 steps，视觉编码器 + 检索器联合训练；best mean R@1=0.184。）

**结论：本次 smoke 的候选检索在 val_unseen 上很弱，尤其是 R@1。** 这不足以支持检索泛化或地图增益结论。可能原因：训练 episode 仅 128 条、80 步远未收敛；候选是通用正射切块而非目标裁剪；语言描述与“目标附近网格块”的跨模态对应本身困难；未加入目标检测/显著性先验。

**Top-3/Top-5 失败处理**：候选不足时零填充，`goal_valid` 记录有效 Goal token 数；Top-5 不足时 map 特征剩余槽位填零。最终 token 缓存中 `goal_valid=3` 的帧占 456/456，但接口已实现并测试了不足时的零填充。

## 3. T5 Token 缓存

| 项目 | 数值 |
|---|---:|
| 帧数 | 456 |
| 轨迹块（train/val_seen/val_unseen） | 48 / 12 / 16 |
| Token 缓存 | 187 MB（本地 `data_cache/`，忽略） |
| GAB 阈值（仅 train 拟合） | 0.0557 |
| train_seen 动作 | Stop 24 / Forward 78 / Right 7 / Left 11 / Up 10 / Down 158 |
| val_seen 动作 | Stop 6 / Forward 9 / Right 1 / Left 2 / Up 0 / Down 54 |
| val_unseen 动作 | Stop 8 / Forward 38 / Right 1 / Left 5 / Up 2 / Down 42 |

历史/位姿严格因果：slot 0 是上一步，slot 15 是 16 步前；缺失槽零填充并 mask。教师动作只在训练中作为标签/过去已执行动作，不使用未来教师动作作为推理输入。

## 4. 小规模 MEC 训练

同一 Token 数据、动作标签、split 和冻结 BERT/视觉编码器；Small / Transformer-M 各训练 full T5+Language 与 map-free baseline。150 steps，batch=8。

| model | retrieval | 参数 | val_seen acc | val_unseen acc | val_unseen move acc | val_unseen Stop P/R |
|---|---:|---:|---:|---:|---:|---:|
| Small | full | 1,007,110 | 0.7500 | 0.4896 | 0.5341 | 0.000 / 0.000 |
| Small | map-free | 1,010,310 | 0.7917 | 0.5000 | 0.5000 | 0.500 / 0.500 |
| Transformer-M | full | 3,232,390 | 0.6806 | 0.4896 | 0.5341 | 0.000 / 0.000 |
| Transformer-M | map-free | 3,393,798 | 0.6944 | 0.4688 | 0.5000 | 0.500 / 0.125 |

- full 与 map-free 的差异没有显示地图检索正增益；本次不能声称地图辅助提升了纯视觉 MEC。
- full 模型的 Arrival 头依赖检索 Goal 匹配特征；检索质量差时 Stop P/R 为 0，是当前最明显的失败模式。
- map-free baseline 去掉了 Goal/GAB/地图证据和 Top-5 分支；它使用 current/history/pose/language 和独立的 current+history+language Arrival 头，因此两者不是完全相同 Arrival 头，报告只作为 Smoke 对照。
- GPU 峰值分配约 90–324 MB（batch=8；另有共享 GPU 进程使 reserved 约 6 GB）；训练 5.9–10.1 s/模型。
- 这是短训练/小样本 Smoke，不能据此判断正式训练后的相对性能。

## 5. 防泄漏测试

新增测试确认：

- 检索推理函数、candidate builder、Goal/Map 特征构建器不接受 target 坐标/ID/crop/teacher action/success；
- T5 Small/Transformer-M 的 `forward_components` 不接受 `target_xyz`、`object_ids`、`teacher_actions` 等字段；
- map-free 模型拒绝 Goal/GAB/map 输入；
- 历史 slot 因果且 mask 正确；
- 候选不足时 Goal/Map 零填充；
- `test_unseen` 未读取。

测试命令：`pytest -q`。

## 6. 未解决与下一步

1. 检索泛化是当前瓶颈：需要更多 train episodes/更长训练、改进候选表示（多尺度视觉/地图上下文/目标性先验），并重新评估 R@1/5/10；
2. 目前候选视觉 token 是 96×96 正射切块的 mean token，未使用对象检测或显著性；仍不得用 GT 目标裁剪；
3. full 模型 Stop 头依赖检索 Goal 匹配；检索失败时需要显式的 `goal_valid`/失败门控，而不是默认信任零填充；
4. 尚未实现正式全量 Token 生成、完整 Small/Transformer-M 训练、val SR/SPL/NE、RL/DQN/PPO 或 GSM；
5. 点云 FPV 空洞、部分地图配准问题仍存在；正式实验前需逐地图几何检查。

## 7. 复现

```bash
python scripts/t5_smoke.py \
  --device cuda \
  --retrieval-steps 80 \
  --retrieval-train-per-map 16 \
  --retrieval-val-per-map 12 \
  --train-trajectories 24 \
  --val-seen-trajectories 6 \
  --val-unseen-trajectories 8 \
  --steps-per-trajectory 6 \
  --mec-steps 150 \
  --mec-batch-size 8 \
  --render-size 336 \
  --crop-radius 120 \
  --candidate-stride 48 \
  --candidate-scales 32 64 128
```
