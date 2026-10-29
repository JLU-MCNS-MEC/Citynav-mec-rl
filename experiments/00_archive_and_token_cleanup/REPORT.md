# 实验 00：旧版归档与 Token 数据集清理

- 日期：2026-10-29
- 分支：`exp/citynav-mec-fixed50m-topdown`（从核验后的 `ebe557f` 创建）
- 状态：**归档、盘点、清单与校验完成**；**Token 数据集删除已就绪，但等待用户明确确认**
  （自动模式对不可逆删除做了权限拦截，见 §3.4）
- 未读取：`test_unseen`

## 1. 旧版核验与归档

### 1.1 提交与分支

| 项 | 值 |
|---|---|
| 参考提交（已核验为 HEAD 祖先） | `ebe557f5e9c5e1b68e918cd1a452d3785d7e9dc9`（2026-10-29 19:55:36，`Add experiment ledger and val_seen closed-loop report`） |
| 归档标签 | `archive/citynav-token-v1-20261029` → 标签对象 `a0eadcca133d351ebcc15c1ac417e491f3613fc2`，peel 到 `ebe557f` |
| 旧分支 | `exp/visual-encoder-ablation-20261028`（保留，仍由实验 02 收尾脚本使用） |
| 核验时的旧分支 HEAD | `e72deb49713cbb084f61d0912c05a3b4d81a1168`（20:47:34 的 live results 提交） |
| 新分支 | `exp/citynav-mec-fixed50m-topdown`，已推送 `origin` |

说明：`ebe557f` 之后旧分支仍在追加实验 02 的实时结果提交；归档标签固定在 `ebe557f`，
即“旧版报告齐全、新方案开始之前”的稳定点。旧分支与旧工作区未被覆盖。

### 1.2 归档位置与内容

服务器归档目录：`/home/20261024_1/Workspace/Citynav-archive/20261029_token_v1/`

| 内容 | 说明 |
|---|---|
| `metadata/t5_tokens.json` | 旧 Token 缓存元数据完整副本（4.7 MB，含 12,800 帧元信息与 GAB 配置） |
| `metadata/token_dataset_stats.json` | 帧数、轨迹数、动作分布、地图列表、14 个数组的 shape/dtype |
| `metadata/token_cleanup_inventory.json` | 删除目标逐项清单（绝对路径、文件数、表观/实际占用、用途） |
| `weights_manifest.csv` + `weights_sha256.txt` | 27 个权重的路径、字节数、SHA-256 |
| `reports_snapshot/` | 旧版 14 份关键报告副本（总览、02/03/06、编码器、闭环、Token 审计） |
| `environment.txt` | 创建时间、源仓库、分支、HEAD、标签、磁盘余量 |

### 1.3 权重清单（27 个文件，均已记录 SHA-256）

覆盖：旧视觉编码器（`runs/visual_encoder_citynav/*/best_encoder.pt`，11 个 run）、
初始化权重 `data_cache/weights/shared_encoder.pt`、检索器
`data_cache/t5_frozen_1000/retrieval/{encoder,retriever}.pt`、
`mec/`、`mec_balanced/`、`mec_sqrt/` 下的 12 个 MEC checkpoint。

与历史报告核对一致的关键 SHA：

| 权重 | SHA-256 |
|---|---|
| E2-FOV-Pose（实验 A 冻结，2751 步） | `441de539f1d30f2be68d77a1c954daef5f3ab3bd3501f33b2c9798f89517011b` |
| E2-FOV-Pose 延长（5502 步） | `8807035e050d86126aba2cd13d6e99c388d719a5fd087427e1ebceb636ed2fb3` |
| 旧基线 `full_20261027_masks_b8` | `322cfa82a06ee926b74541ecde30f7ecd20b15c0a6ddd687647a0cd80d27b23b` |

原始权重仍保留在 `runs/` 与 `data_cache/t5_frozen_1000/{mec,mec_balanced,mec_sqrt,retrieval}/`，
本阶段的清理目标**不包含**任何权重、报告、原始数据或源码。

## 2. Token 数据集盘点

### 2.1 CityNav-MEC 主 Token 缓存

路径：`/home/20261024_1/Workspace/Citynav-mec-rl/data_cache/t5_frozen_1000/tokens/`

| 文件 | 文件数 | 表观大小 | 实际占用 |
|---|---:|---:|---:|
| `t5_tokens.npz` | 1 | 5.29 GB | 5.29 GB |
| `shards/` | 1,600 | 5.33 GB | 5.33 GB |
| `t5_tokens.json` | 1 | 4.7 MB | 4.7 MB |
| **合计** | **1,602** | **10.63 GB** | **10.63 GB** |

数据集规格（来自元数据与 npz 头）：

| 项 | 值 |
|---|---|
| 帧数 | 12,800 |
| 轨迹（窗口）数 | train_seen 1,250 / val_seen 70 / val_unseen 280 |
| 覆盖地图 | 12 张：birmingham_block_{5,10,11,12,13}、cambridge_block_{2,3,4,10,17,23,25} |
| 动作分布（train） | Stop 625、MoveForward 2,965、TurnRight 493、TurnLeft 532、GoUp 372、GoDown 5,013 |
| GAB 阈值 | 0.03880675001548639 |
| 数组 | `current_tokens (12800,576,384) fp16`、`goal_tokens (12800,3,384) fp32`、`history_vis (12800,16,768) fp16`、`history_pose (12800,16,12)`、`history_action (12800,16,6)`、`history_mask (12800,16)`、`gab (12800,20)`、`map_feat (12800,22)`、`language_features (12800,768)`、`actions`、`split_id`、`step_id`、`goal_valid`、`top5_indices (12800,5)` |
| `test_unseen_read` | false |

### 2.2 Token 对齐审计派生缓存

路径：`experiments/visual_encoder_citynav/token_alignment_audit_20261029/cache/`
（6 个 arm：probe_h0 / probe_h0_pretrain / probe_h16 / subset_h0 / subset_h0_pretrain / subset_h16）

本阶段只删除其中的 **Token 载荷**：`tokens/t5_tokens.npz`、`tokens/t5_tokens.json`、
`parts/*.npz`；**保留**同一目录下的检索权重、MEC 权重与语言特征（不属于 Token 数据集）。

| 内容 | 文件数 | 大小 |
|---|---:|---:|
| Token npz（6 个 arm） | 6 | 2.20 GB |
| Token json（6 个） | 6 | < 1 MB |
| `parts/*.npz`（2 个 arm） | 5 | 0.71 GB |
| **合计** | **17** | **2.91 GB** |

### 2.3 其他项目扫描

| 位置 | 结果 |
|---|---|
| `/home/20261024_1/Workspace/mec-v8.2` | 只有生成脚本（`generate_*_token_dataset.py`、`token_data.py`），**没有本地 Token 数据集** |
| 全盘 `token_dataset_*` | **0 个目录** |
| 其他 Token 目录 | 只有上述两处；未发现 `token_dataset_train/val/test` |

## 3. 删除清单与状态

### 3.1 目标（共 20 项、1,619 个文件）

| 类别 | 路径 | 大小 |
|---|---|---:|
| 核心 | `data_cache/t5_frozen_1000/tokens/t5_tokens.npz` | 5.29 GB |
| 核心 | `data_cache/t5_frozen_1000/tokens/shards/` | 5.33 GB |
| 核心 | `data_cache/t5_frozen_1000/tokens/t5_tokens.json` | 4.7 MB |
| 审计 | 6 个 arm 的 `tokens/t5_tokens.npz` | 2.20 GB |
| 审计 | 6 个 arm 的 `tokens/t5_tokens.json` | < 1 MB |
| 审计 | `subset_h0_pretrain/parts/part_0{0..3}.npz` 与 `probe_h0_pretrain/parts/part_00.npz` | 0.71 GB |
| **合计** | 20 项 | **13.49 GB（12.57 GiB）** |

### 3.2 明确不删除

原始 CityNav JSON、PLY、`DATA/rgbd` 的 PNG/TIF、源码、配置、实验报告、日志、
视觉编码器权重、检索器权重、全部 MEC checkpoint、归档目录本身。

### 3.3 删除前的安全检查

- 对两个 Token 目录执行 `lsof +D`：**无进程持有句柄**；
- 正在运行的实验 02 闭环只使用 `t5_tokens.json` 的 GAB 阈值（启动时已读入内存），
  不读取 `t5_tokens.npz` 或 shards；本报告写作时旧实验仍在运行，删除不影响其收尾；
- 清理脚本先写清单/校验和，再逐项删除，不使用宽泛通配符。

### 3.4 当前状态：等待用户确认

执行阶段被自动模式权限拦截（不可逆删除需要用户的明确指令）。归档与清单已全部落盘，
删除可随时执行。复现命令：

```bash
cd /home/20261024_1/Workspace/Citynav-mec-rl-fixed50m
$HOME/miniconda3/envs/DRL_migrated/bin/python \
  experiments/00_archive_and_token_cleanup/scripts/archive_and_cleanup.py --execute
```

脚本执行后会写入 `metadata/token_cleanup_result.json`：逐项删除记录、删除前/后可用空间、
实际释放字节、残留目标扫描，并自动删除空的 `tokens/` 目录。

## 4. 删除后核验（待执行）

执行后需补齐：

1. `df` 前后差值 vs 表观 13.49 GB；
2. 残留扫描（20 个目标全部不存在；`token_dataset_*` 仍为 0）；
3. `du -sh data_cache/t5_frozen_1000`（预期从 11 GB 降到约 0.3 GB：`mec*`、`retrieval`、
   `language_features.npz` 保留）。

## 5. 结论与下一步

- 旧版归档已固定：标签 `archive/citynav-token-v1-20261029`、权重清单、报告快照、
  Token 元数据副本均在服务器归档目录，旧分支保留；
- 旧 Token 数据集已完整盘点（13.49 GB），删除脚本与清单齐备，**只差用户确认**；
- 确认并删除后，进入实验 01：俯视地图资产审计（`DATA/rgbd` 已有 34 张地图的
  North-up PNG + 高程 TIF，共 3.4 GB；官方 `rasterize.py` / `cropclient.py` 需从
  官方仓库拉取核对）。
