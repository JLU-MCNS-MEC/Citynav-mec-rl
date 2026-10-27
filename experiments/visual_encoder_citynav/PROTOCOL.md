# CityNav 共享视觉编码器预训练协议

## 1. 范围

本实验只训练 `VisualTokenEncoder`：

- 输入 336×336 RGB，24×24 patch，输出 576×384 token；
- 可从 ToyEnv V8.2 `shared_encoder.pt` 初始化；
- 训练完成后只保存 encoder 权重；辅助动作/跨视角/检索头不进入最终模型；
- 不训练 Small/Transformer-M MEC，不生成正式 MEC Token 数据集，不开发 RL。

## 2. 数据与划分

| split | 官方轨迹数 | 本实验用途 |
|---|---:|---|
| Train-Seen | 22,002 | 正式训练数据池（每条轨迹取一个教师步样本，单 epoch 覆盖全部） |
| Val-Seen | 2,498 | 固定 256 条子集用于 Checkpoint 选择；完整数量记录在报告中 |
| Val-Unseen | 2,826 | 固定 256 条子集用于独立泛化诊断，**不参与选模** |
| Test-Unseen | 5,311 | 封存，不读取 |

- 官方划分保持不变；
- 原始 JSON/CityRefer/RGB/TIFF/PLY 只读；
- 点云 downsample 缓存、候选库、manifest、语言特征、checkpoint、TensorBoard
  均写入本项目 `data_cache/` / `runs/`，不提交。

训练 manifest 的每条轨迹选一个教师动作步；动作采样按六类循环选择，若该轨迹
没有目标类别则保留实际标签回退中段步。六类实际数量由 `class_weights`
反频率加权，**不修改标签**。

## 3. 输入与无泄漏约束

- 当前 RGB：SensatUrban PLY 第一视角渲染；
- 语言：冻结 `bert-base-uncased` 768 维；
- 地图候选：官方正射图固定网格（stride 48 m）+ 固定尺度（32/64/128 m）
  的 96×96 切块，不使用 CityRefer 目标框/ID/target 坐标生成；
- **禁止**使用 Ground Truth 目标裁剪作为输入；
- 当前位姿只用于构造 cross-view 正样本 mask；
- `target_positions` 只用于训练阶段构造 retrieval 正负标签；
- 不把当前位姿匹配当成目标位置匹配：两个 mask、两个投影头、两个指标分开；
- 推理/验证排序只依赖语言特征与候选视觉特征，不接受 target 坐标/ID/crop/
  teacher action/success；
- 教师动作只作为动作标签与过去已执行动作，不作为未来输入；
- `test_unseen` 永不读取。

## 4. 三个训练目标

### L_action（辅助动作头）

- 当前视觉 mean+max [768] + 冻结语言 [768] → 临时 MLP → 六动作 logits；
- 加权交叉熵，类别权重按 Train manifest 实际频率反比；
- 只给视觉编码器提供导航相关信号，不是正式 MEC。

### L_xview（当前位姿跨视角匹配）

- query：当前 FPV 视觉全局特征；
- candidates：同一区块固定网格候选；
- 正样本：候选中心距**当前 UAV 位姿** ≤30 m；
- 负样本：同区块其他候选，优先采样 30–120 m 的困难负样本；
- 多正样本 InfoNCE（temperature 0.07）；与目标位置无关。

### L_retrieval（语言→目标候选检索）

- query：冻结 BERT 语言特征；
- candidates：同区块固定网格候选；
- 正样本：候选中心距 `target_positions[-1]` ≤40 m（仅训练标签）；
- 负样本：同区块困难负样本；
- 多正样本 InfoNCE；验证时对全候选库排序，报告 Recall@1/5/10；
- 该目标与 `L_xview` 的 mask/头/指标严格分开。

总损失：

```text
L = L_action + 0.5 * L_xview + 0.5 * L_retrieval
```

权重在严格分离 mask 后的 Smoke 中根据各损失数值尺度与 encoder 梯度范数
（action 1.69 / xview 1.30 / retrieval 1.71）确定，两个对比损失各取 0.5，
避免辅助匹配梯度压过动作监督。

## 5. 训练设置

- Train-Seen 全部 22,002 条轨迹，每条取一个教师步样本；
- 1 个 epoch；batch 8（首轮 batch 32 在共享 GPU 上 OOM，数据池/epoch/损失不变，
  见资源报告）；AdamW；
  - encoder lr 1e-5，辅助头 lr 1e-3，weight decay 1e-4；
  - cosine LR，最低为 encoder lr 的 10%；
  - grad clip 1.0；
- 4M 点云 downsample 缓存；336 第一视角；候选 96×96；
- Checkpoint 在 25/50/75/100% 更新步；
- 选模只看 Val-Seen 的固定 256 子集：
  `score = 0.4*macro_F1 + 0.3*cross_view_top1 + 0.3*retrieval_recall@5`；
- Val-Unseen 独立记录，不参与 best checkpoint 选择；
- 最佳/最终 checkpoint 只保存 encoder state_dict，并记录 SHA-256。

## 6. 必报指标

- 总损失、action/xview/retrieval 独立训练损失；
- Val-Seen/Val-Unseen 的 action accuracy、macro-F1、六类分别 accuracy；
- cross-view Top-1/Top-5；
- retrieval Recall@1/5/10；
- encoder 梯度范数、参数变化量、学习率；
- render 时间、FPV 有效像素比例、显存、吞吐率；
- 各 split 样本量/动作数量/地图覆盖；
- 按地图分组的 action/retrieval/valid-pixel 结果；
- 初始化权重 SHA/加载比例、最终 encoder SHA/参数变化；
- 辅助头不进入最终 checkpoint 的检查。

## 7. 资源与已知限制

- Smoke（4M 点云、336、batch 8）实测：FPV 有效率约 0.40，渲染约 0.14 s/帧；
- 全量 Train-Seen 约 22k 第一视角帧，预计渲染 50–90 分钟（取决于 worker 与
  缓存命中），GPU 训练与验证另计；
- 点云 downsample 会造成空洞，不能通过修改原始数据或目标标签规避；
- 若空洞进一步限制训练，优先研究更高密度/表面重建，而不是引入目标监督泄漏；
- 不启动 Small/Transformer-M、正式 Token 数据集或 RL。
