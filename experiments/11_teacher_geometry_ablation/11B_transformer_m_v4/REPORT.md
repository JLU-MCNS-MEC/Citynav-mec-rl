# 实验 11B 报告：M-v4 纯教师监督训练（Teacher-only Transformer-M）

- 状态：**训练中**（2026-11-02 启动）
- 目标：在严格纯教师监督协议下训练 Transformer-M v4（seed42），作为 M-v5 多地标几何消融的公平基线。

## 1. 数据（11A 最终池，纯教师）

| 项 | 值 |
|---|---|
| 训练 Episode | 18,521（Stage 02 High+Medium ∩ Stage 03 四动作教师成功） |
| 训练决策状态 | 1,279,882（全量，无每集上限、无重采样、无混合） |
| 验证集 | Val-Seen 400 集 × 4 状态 = 1,574 状态（Stage-03 式四动作教师标签，**仅验证用**） |
| 特征来源 | `build_features_11b.py` 因果回放（C1/09C 计算与 10C 相同；轨迹 = Stage 03 四动作教师，**非 03B**） |
| 存储 | memmap `.npy`（~85 GB）+ 分块 DataLoader（文档 4.3） |

数据红线（文档 2.x）：无 03B 轨迹、无 C1 闭环状态/动作、无 Val/Test 教师标签进训练、无真实目标坐标/距离/未来位姿/成功标志作为输入。M-v3 权重与 250k 混合样本完整保留作历史对照，未复用。

## 2. 模型与三处修复（v3 → v4）

| 修复 | v3 问题（代码定位） | v4 实现 |
|---|---|---|
| ① 历史时间位置编码 | 16 个历史槽共享同一 type embedding，模型无法区分"1 步前"与"16 步前" | 可学习 `hist_time` (16,256)，按槽位加到历史 token |
| ② 重复特征 | 融合状态 15 维中 dx/dy/score 各出现两次（belief[8:10]==match[0:2]，belief[4]==match[2]） | 去重为 12 维：belief[fwd,lat,dist,bearing,unc,ent,since] + match[dx,dy,score,margin,sharpness] |
| ③ 训练调度 | OneCycleLR 按 epoch 步进（total_steps=30，非常规） | 按 batch 步进（total_steps=epochs×steps_per_epoch）+ NaN 防护 |

其余架构不变：d_model 256 / 4 层 / 8 头 / FFN 1024；输入契约同 M-v3（当前 7×7×2048 空间视觉 + C1 目标描述子 512 + 16 步因果历史 + 09C 信念/C1 匹配融合状态）。

## 3. 训练协议

| 项 | 值 |
|---|---|
| 超参 | batch 128、epochs 30（early-stop patience 10）、lr 3e-4、AdamW（wd 0.01）、OneCycleLR warmup 0.03、梯度裁剪 1.0 |
| 种子 | seed 42（仅此一个种子，按文档要求） |
| 当前视觉帧 | 训练时从 rgbd 栅格实时裁剪 + 冻结 ResNet-50 layer4 编码（同 10D） |
| 模型选择 | 每 epoch Val-Seen Macro-F1 选 best.pt，patience 10 |
| 输出 | `runs_11b/seed42/`（best.pt、epoch_NN.pt、checkpoint.pt、history.jsonl、config.json、train_progress.json、gpu_util.jsonl） |

## 4. 训练结果（占位，完成后填写）

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Macro-F1 | Stop Rec | Turn-L/R Rec | LR | sec |
|---|---|---|---|---|---|---|---|---|---|
| … | … | … | … | … | … | … | … | … | … |

- 最终 best.pt：epoch __，Val Macro-F1 __
- 训练总时长：__

## 5. 与 M-v3 的可比性说明

- 数据协议不同（v3=250k 混合含 03B 轨迹+50k C1 恢复状态；v4=1.28M 纯 Stage 03 四动作教师状态），**Val 指标不可直接对比**；公平对比以 11D 闭环 SR 为准。
- 验证集监督不同（v3 用 03B val 标签，v4 用 Stage-03 式四动作 val 标签），故两版 val Macro-F1 度量口径不同，仅用于各自模型选择。

## 6. 产物与推送

- 代码：`11B_transformer_m_v4/scripts/`（build_features / build_val_teacher / materialize / dataset / model / train）
- 大产物（gitignore）：`results/cache/`（特征缓存）、`results/data/`（memmap）、`runs_11b/*.pt`
- 入库产物：history.jsonl、config.json、train_progress.json、gpu_util.jsonl、本报告
