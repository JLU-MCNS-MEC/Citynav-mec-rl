# 实验 11A 报告：纯人工教师数据集审计

- 状态：**完成**
- 日期：2026-11-01
- 目标：确定 M-v4 / M-v5 的纯教师监督训练池（Stage 02 High+Medium ∩ Stage 03 四动作教师成功），完成十项审计统计，产出清单文件。

## 1. 输入与来源

| 输入 | 来源 | SHA256（前 16 位） |
|---|---|---|
| `citynav_train_seen.json`（22,002 条原始人工轨迹） | 官方 CityNav 数据（与旧服务器原始文件逐字节一致，迁移清单校验通过） | `bfaabed32d56aad7…` |
| `selection_high_medium_mixed.json`（18,521 条） | Stage 02 质量审计（仓库） | 见 `teacher_pool_audit.json` |
| `train_quality.csv`（22,002 行逐条指标） | Stage 02 | 同上 |
| `teacher_episodes.csv`（22,002 行） | Stage 03 再生成（与 git 历史版本零字节差异） | 同上 |
| `four_action_labels.npz` | Stage 03 再生成（确定性，36 s 全量） | 同上 |

## 2. 交集计算

```
最终训练池 = Stage 02 High+Medium (18,521)  ∩  Stage 03 success=true (19,105)
```

独立核对（train_quality.csv ↔ teacher_episodes.csv ↔ 筛选文件三方互验）：

| 集合 | 数量 |
|---|---|
| High | 14,237 |
| Medium | 4,284 |
| High + Medium | 18,521 |
| 四动作教师成功 | 19,105 |
| High+Medium 但不成功 | **0** |
| 成功但未达质量线（Low，被排除） | 584 |
| **最终训练池** | **18,521**（与 18,521 巧合相等，系重新计算核实，非直接采用） |

排除 Low 质量（3,481 条）后，最终池恰好全部通过四动作教师回放。

## 3. 十项审计统计

### 3.1 最终 Episode 数：**18,521**

### 3.2 完整决策状态数：**1,279,882**（均值 69.1 状态/条）

### 3.3 每张地图 Episode / 状态数（24 张训练地图，按状态数降序）

| 地图 | Episodes | 状态数 | 地图 | Episodes | 状态数 |
|---|---|---|---|---|---|
| birmingham_block_1 | 2,471 | 170,059 | cambridge_block_4 | 263 | 10,296 |
| cambridge_block_20 | 1,635 | 127,153 | cambridge_block_23 | 262 | 7,697 |
| cambridge_block_13 | 1,430 | 117,605 | cambridge_block_25 | 139 | 5,680 |
| cambridge_block_9 | 1,237 | 101,124 | birmingham_block_13 | 29 | 260 |
| cambridge_block_21 | 1,425 | 97,662 | birmingham_block_11 | 15 | 237 |
| cambridge_block_7 | 1,055 | 90,608 | birmingham_block_12 | 707 | 29,740 |
| birmingham_block_3 | 1,320 | 86,130 | cambridge_block_19 | 294 | 27,112 |
| cambridge_block_28 | 1,198 | 85,481 | cambridge_block_18 | 425 | 20,715 |
| cambridge_block_12 | 1,163 | 81,558 | cambridge_block_6 | 241 | 16,465 |
| birmingham_block_9 | 773 | 63,094 | cambridge_block_17 | 314 | 16,333 |
| birmingham_block_4 | 766 | 60,902 | birmingham_block_10 | 257 | 12,912 |
| cambridge_block_33 | 780 | 38,578 | birmingham_block_7 | 322 | 12,481 |

### 3.4 Short / Medium / Long 分布（Stage 03 全局三分位：44 / 77 步）

| 长度组 | Episodes | 占比 | 状态数 | 状态占比 |
|---|---|---|---|---|
| Short（≤44 步） | 6,225 | 33.6% | 164,665 | 12.9% |
| Medium（45–77 步） | 6,246 | 33.7% | 377,456 | 29.5% |
| Long（>77 步） | 6,050 | 32.7% | 737,761 | 57.6% |

长任务不再被压缩：Long 组贡献了 57.6% 的决策状态。

### 3.5 轨迹长度分布（steps，最终池）

| 分位 | p5 | p25 | p50 | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| 步数 | 13 | 36 | 60 | 87 | 125 | 161 | 255 |

### 3.6 四类动作比例（1,279,882 个决策状态）

| 动作 | 数量 | 占比 |
|---|---|---|
| Forward | 946,031 | 73.9% |
| TurnRight | 158,166 | 12.4% |
| TurnLeft | 157,164 | 12.3% |
| Stop | 18,521 | 1.4% |

左右转比例接近均衡（0.6% 偏差），**未做任何人工重采样**；每条成功轨迹恰好 1 个 STOP。

### 3.7 目标距离分布（start_target_m）

| 分位 | p5 | p25 | p50 | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| 米 | 44.2 | 114.9 | 187.8 | 266.6 | 331.3 | 363.4 | 417.6 |

### 3.8 路径效率（Stage 02 定义：起始目标距离/路径长度）

中位 0.742，均值 0.688；p90 = 1.00，p99 = 1.09（个别轨迹路径略短于直线距离，系固定高度折线折算所致）。

### 3.9 回退比例（backtrack_m / path_len_m）

中位 3.6%，均值 7.5%，p90 = 21.9%；回退比例 >10% 的轨迹共 5,871 条（31.7%）。

### 3.10 四动作教师回放成功率

| 范围 | 成功率 |
|---|---|
| 全部 Train-Seen（22,002 条） | 19,105 / 22,002 = **86.83%** |
| 最终训练池（18,521 条） | 18,521 / 18,521 = **100%**（构造保证） |

## 4. 清单文件（4.4）

| 文件 | 内容 | 大小 |
|---|---|---|
| `teacher_episode_manifest.csv` | 18,521 行；index、key、map、quality、steps、length_group、起止距离、效率、回退、转弯、fixed_z 等；**保留原始轨迹 ID 与质量标签** | 3.7 MB |
| `teacher_state_manifest.csv` | 1,279,882 行；state_id、ep_index、key、map、t、action、length_group（全量状态池，未设每集上限、未重采样） | 94.6 MB |
| `teacher_pool_audit.json` | 全部统计 + 输入文件哈希 | 4.2 KB |

## 5. 数据规模与资源评估（对应 4.3）

- 全量状态池：**1,279,882 状态**（Stage 10 M-v3 的 200,000 教师状态仅为其中 15.6%）
- 特征物化估算（hist_vis f16 64KB + hist_pose/hist_act/hist_mask + belief/match/target_emb/states/labels ≈ 66 KB/状态）：
  - 磁盘：**≈ 85 GB**（memmap `.npy`，`/mnt/hdd2t` 剩余 1.7 TB，充足）
  - RAM：62 GB 无法整载 → **按文档要求采用 memmap + 分块 DataLoader**（10D 的"一次性载入 RAM"方案不可沿用）
- 特征来源：**复用 Stage 10C 缓存**（79 GB，含全部 22,002 条 episode 的全步 C1 特征；已核对结构完整），无需重跑因果回放
- 训练时当前视觉帧：与 10D 相同，从 rgbd 栅格按状态实时裁剪 + 冻结 ResNet 编码（HDD IO 瓶颈，10D 实测可接受）

## 6. 结论

- 最终纯教师训练池：**18,521 条 episode / 1,279,882 个决策状态**，数据链路完全来自原始 CityNav 人工轨迹（Stage 02 筛选 → Stage 03 教师转换），无任何 03B / 10C 控制器 / 验证集数据混入。
- 长任务（Long 组）保留完整，贡献 57.6% 的状态；Short/Medium/Long 三组均近 1/3，避免 High-only 的长轨迹偏差。
- 所有中间产物均可由原始数据 + 仓库代码确定性再生（本报告输入哈希已记录）。
