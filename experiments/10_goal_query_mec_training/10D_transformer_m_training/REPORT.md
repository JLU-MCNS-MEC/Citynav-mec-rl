# 10D 报告：Transformer-M v3 正式训练（seed 42，唯一一次）

- 日期：2026-10-31
- 服务器：新服务器 `ubuntu`（RTX 5090 32 GB）
- 分支：`exp/citynav-mec-fixed50m-topdown`
- 输入契约：`../10C_mec_input_dataset/INPUT_CONTRACT.md`
- 模型：`scripts/model10c.py`（TransformerM v3，4,492,804 参数）

## 1. 设置

| 项 | 值 |
|---|---|
| 训练数据 | 250,000 决策状态：Train-Seen 教师 200,000（逐状态均匀）+ C1 恢复状态 50,000（DAgger 式，20%，标签=C1 动作） |
| 验证（每 epoch） | Val-Seen 固定子集 1,564 状态（400 episode × ~4），不参与训练 |
| 模型 | 4 层 Transformer，d=256，8 头，ffn=1024；CLS + 49 视觉 token + 目标 + 16 历史 + 状态 token |
| 优化 | AdamW lr=3e-4，wd=0.01，OneCycleLR（warmup 3%），batch 128，grad clip 1.0 |
| 损失 | **四类交叉熵，无类别加权 / pos_weight / focal**（用户禁令遵守） |
| 停止 | 最大 30 epoch；Val Macro-F1 无提升 patience 10 → **early stop 于 epoch 16**（best=epoch 5） |
| 种子 | 42（只训练这一次） |

## 2. 结果

### 2.1 最佳 checkpoint（epoch 5）

| 指标 | 值 |
|---|---|
| Val Loss | 0.1461 |
| Val Accuracy | 95.2% |
| **Val Macro-F1** | **0.9228**（旧版 Stage 05 最佳 0.784，**+0.139**） |
| Train Loss / Acc（同 epoch） | 0.0937 / 96.8% |
| STOP F1（P/R） | 0.932（0.923 / 0.941） |
| Forward F1（P/R） | 0.969（0.961 / 0.976） |
| TurnLeft F1（P/R） | 0.894（0.906 / 0.882） |
| TurnRight F1（P/R） | 0.896（0.936 / 0.860） |

### 2.2 混淆矩阵（Val-Seen，best epoch）

```
            pred  Stop  Fwd  TL  TR
true Stop           48    3   0   0
true Forward         4 1146  14  10
true TurnLeft        0   17 135   1
true TurnRight       0   26   0 160
```

### 2.3 训练动态

- **Train Acc 未出现过快到 1**：epoch 0 即 95.8%（belief≈带噪 GPS 的任务简化所致，
  Stage 07 已记录该性质），best epoch 96.8%，末 epoch 98.5%——末期出现轻微
  过拟合迹象（train acc 升、val macro_f1 从 0.923 缓降至 0.896），early
  stopping 正确截断。
- 训练速度：每 epoch ~128-207 s（RAM 物化特征 + 批内现算当前视觉），
  16 epoch 共 **2,063 s（34 分钟）**；Stage 05 的 30 epoch 为 80 分钟。
- GPU：显存 ~1.8 GB；epoch 末快照平均 util 5.1%（快照在验证之后采集，
  低估真实训练期利用率；训练期内利用率见训练曲线图）。
- 损失曲线 / 混淆矩阵：`figures/10d_training_curves.png`、`10d_confusion.png`。

### 2.4 checkpoint 保存（用户要求：保存相关模型权重）

| 文件 | 说明 | SHA-256（前 20） |
|---|---|---|
| `results/runs_10d/seed42/best.pt` | 最佳（Val Macro-F1） | `5f01a52dd5bbb53e361a` |
| `results/runs_10d/seed42/epoch_00..15.pt` | 每 epoch 快照 | 见目录 |
| `results/runs_10d/seed42/checkpoint.pt` | 断点续跑状态（含优化器/RNG） | 见目录 |
| `results/runs_10d/seed42/history.jsonl` | 每 epoch 全指标 | — |

权重不入库（仓库 *.pt gitignore 惯例），路径与 SHA 已在此记录。

## 3. 结论

- 新输入模态（C1 目标描述子 + 09C 连续信念 + 恢复状态 + 修正采样）下，
  Transformer-M 的离线 Val Macro-F1 达到 **0.923**，显著高于旧版 0.784；
  STOP 召回从旧版瓶颈（0.485）升到 **0.941**。
- 是否存在过拟合：末期轻微（+2pp train acc，val 缓降），early stopping 生效。
- **闭环是否真正变好，以 10E 为准**（离线指标不能替代 SR）。

## 4. 复现

```bash
export CITYNAV_DATA=/mnt/hdd8t/Citynav-MEC/DATA
cd experiments/10_goal_query_mec_training/10D_transformer_m_training
python -I scripts/materialize_10d.py --split val
python -I scripts/materialize_10d.py --split train
python -I scripts/train_10d.py --seed 42 --epochs 30   # --resume 断点续跑
python -I scripts/make_figures_10d.py
```

## 5. 运行记录

- 两轮失败：① HDD 随机读（episode npz 逐样本加载）→ 物化为 16GB RAM 张量；
  ② belief 后验熵 -inf（09C 单分量混合 log(1)=0）→ 特征侧保护为 0
  （09C 历史代码未动，物化数组就地修补 257 个值）；③ np.savez zip 4GB 条目
  缺陷 → 逐数组 .npy 目录。
