# 视觉编码器四组消融实验协议（2026-10-28）

本文件锁定 E1–E4 的共同协议与唯一差异。任何偏离都必须写进实验报告，不能静默修改。

## 1. 范围

- 只训练 `VisualTokenEncoder`（336×336 输入，24×24 patch，576×384 token）。
- 不训练 Small / Transformer-M，不生成 MEC Token 数据集，不做 RL。
- 辅助头（动作、跨视角、检索）只用于预训练，最终只保存 encoder。
- Test-Unseen 全程封存，不读取、不下载其 6 个地图区块。

## 2. 固定项（四组完全一致）

| 项 | 值 |
|---|---|
| 随机种子 | 42 |
| 初始化权重 | `data_cache/weights/shared_encoder.pt`（同一份文件，四组共用） |
| 数据划分 | 官方 `original_citynav`，Train-Seen 训练 / Val-Seen 选模 / Val-Unseen 仅诊断 |
| 更新步数 | 2751（`total_steps`，不随采样数变化） |
| batch size | 8 |
| 损失 | `L = L_action + 0.5 * L_xview + 0.5 * L_retrieval` |
| 优化器 | AdamW，encoder lr 1e-5，辅助头 lr 1e-3，weight decay 1e-4 |
| 调度 | cosine，最低 10% encoder lr |
| 梯度裁剪 | 1.0 |
| 选模 | Val-Seen：`0.4*macro_F1 + 0.3*xview_top1 + 0.3*R@5` |
| Val 子集 | 固定 256 条（Val-Seen / Val-Unseen 各 256） |

## 3. 四组唯一差异

| 组 | 标签审计 | 渲染 | 教师步采样 | 配置 |
|---|---|---|---|---|
| E1 | 已审计（见第 4 节） | 基线 `splat_radius=0`、`max_points=4M` | 每轨迹 1 步 | `configs/e1.yaml` |
| E2 | 同 E1 | `splat_radius=1`、`max_points=8M` | 每轨迹 1 步 | `configs/e2.yaml` |
| E3 | 同 E1 | 基线 | 每轨迹 3 步 | `configs/e3.yaml` |
| E4 | 同 E1 | `splat_radius=1`、`max_points=8M` | 每轨迹 3 步 | `configs/e4.yaml` |

### E1：审计后基线

审计脚本 `audit_labels.py`，只读，不改数据。检查：

1. 方向向量约定：`yaw = atan2(dy, dx)` 与相邻点运动方向；
2. 左右转符号：`TURN_LEFT` 必须对应世界坐标下的逆时针（叉积 > 0）；
3. 升降符号：`GO_UP/GO_DOWN` 对应演示轨迹的 z 变化方向；
4. 前进动作：水平运动且转向角在阈值内；
5. `marker_positions[-1]`（教师停止点）与 `target_positions[-1]`（评估目标）不得混用；
6. 动作执行后位姿必须等于官方动作表的一步。

**只有可证实的错误才允许修复；未发现错误则保留原始标签。**

### E2：渲染优化

- `max_points` 4M → 8M：真实点云更密，重建缓存 `data_cache/visual_encoder_citynav/clouds/`（不覆盖 4M 缓存）。
- `splat_radius` 0 → 1：每个真实点覆盖 3×3 像素，只加宽已有点，不为空区域合成几何。
- 必须报告：有效像素比例、渲染耗时、显存、吞吐，以及空洞率变化。
- 禁止：用目标标签、修改原始数据、基于 TIF 反推点云来"补洞"。

### E3：时序与动作采样

- 每条轨迹选 **3 个教师步**：
  1. 第 1 步 = E1 的基线步（保证可比）；
  2. 其余来自动作切换点（`a[i] != a[i-1]`）与六类动作在轨迹前/中/后三分段的分布，确定性选取。
- 全部来自官方教师 rollout，不合成、不改标签。
- 验证集保持自然分布（每轨迹 1 步），不做类别均衡。
- `total_steps` 固定 2751，因此训练看到的样本总数仍约 22k，但每次抽取的动作分布改变。
- 必须报告 manifest 实际行数、六类动作计数，以及类别权重（由该 manifest 反频率计算，随采样方案变化——这是采样设计的一部分，不是 lambda 变更）。

### E4：E2 + E3

## 4. E1 审计结果（2026-10-28，Val-Seen 全量 2,498 条）

- 六类动作与世界坐标运动向量的独立几何一致性（用运动向量叉积复检，不依赖方向向量）：
  - **全部 100%**：TurnLeft 23,796/23,796、TurnRight 24,353/24,353、
    GoUp 76,652/76,652、GoDown 189,916/189,916、MoveForward 154,032/154,032、Stop 1,578/1,578
- 教师动作执行后位姿与官方动作表逐步一致：**0 处不匹配**。
- `marker` 与 `target` 中位距离 6.46 m、p95 24.18 m、最大 29.97 m，确认二者不可混用。
- 方向向量与局部运动方向：中位偏差 ≈0°（1.4e-7°），但约 12% 的相邻点对偏差 >90°，
  且 777 条轨迹存在方向向量为零的点（代码回退到运动方向）。这是"朝向 vs 速度"的正常
  差异（转向、折返），不是标签错误；用运动向量复检后动作符号仍 100% 一致。
- 原始动作计数（Val-Seen 全部教师步）：MoveForward 154,406、GoDown 190,354、
  GoUp 76,674、TurnRight 24,405、TurnLeft 23,850、Stop 2,498。转向与上升样本明显偏少，
  这是 E3 采样优化的依据。
- 结论：**未发现可证实的坐标或标签错误，不修改标签。**

结果：`results/e1_audit_val_seen_summary.json`（汇总）与 `e1_audit_val_seen_200.json`
（200 条逐条明细）。2,498 条逐条明细体积 2.2 MB，保存在 `data_cache/`，不入库。

## 5. 已知差异与限制

1. **种子**：旧基线 `full_20261027_masks_b8` 用 seed 0，本轮用 42。E1–E4 内部可比；与旧基线比较时混入种子效应，报告中必须分开陈述。
2. **初始化权重**：旧基线的 `shared_encoder.pt`（SHA `8ee6194e...`）未随仓库发布，本轮由 `mec-v8.2/train_shared_encoder.py` 重新生成；SHA 与旧文件是否一致以实测为准，并在报告中标注。
3. **数据覆盖**：HF 镜像只覆盖 34 个开发区块中的 31 个（Train-Seen/Val-Seen 缺 cambridge_block_6/7/9）。缺口对训练池的影响（14.42%）必须在正式训练前报告并由用户确认处理方式。
4. **类别权重**：由各 manifest 反频率计算，E3/E4 的权重会随采样分布变化；报告中同时给出权重和样本数。

## 6. 产物与同步

- 运行目录：`runs/visual_encoder_citynav/ablations/20261028/{e1,e2,e3,e4}_seed42/`
- 缓存：`data_cache/visual_encoder_citynav/`（不入库）
- 结果：`experiments/visual_encoder_citynav/ablations/20261028/results/`
- 图：`.../figures/`
- 分支：`exp/visual-encoder-ablation-20261028`
- 只提交代码、配置、协议、汇总 CSV/JSON、图和中文报告；不提交权重、PLY、原始数据、完整日志。
