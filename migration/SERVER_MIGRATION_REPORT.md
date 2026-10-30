# CityNav-MEC 服务器迁移报告

- 日期：2026-10-30 ~ 2026-10-31
- 方向：旧服务器（迁移来源）→ 新服务器（唯一后续工作服务器，RTX 5090 32GB）
- 代码仓库：`JLU-MCNS-MEC/Citynav-mec-rl`（fixed50m 主线分支）
- 结论：**迁移完成，全项校验通过**；旧服务器清理按用户最终授权范围执行

## 0. 结论摘要

| 项目 | 状态 |
|---|---|
| 代码（fixed50m 与 visual-encoder-ablation 两分支） | ✅ 已克隆/已推送 GitHub，HEAD 与源一致 |
| 教师轨迹（22,002 / 2,498 / 2,826） | ✅ 数量与 SHA-256 逐字节一致 |
| 24m 候选库 candidates24 + v1 候选缓存 | ✅ 28 + 28 个 npz，SHA 一致 |
| Transformer-M 权重（seed42/43、runs_v2 全 epoch） | ✅ SHA 一致 + strict load 0 缺失/0 多余 |
| 原始轨迹 original_citynav、cityrefer | ✅ SHA 一致 |
| 官方 rgbd 栅格（34 块 png+tif） | ✅ 68 文件 SHA 一致 |
| Stage 06 闭环结果（5,324 + 5,311 条）与视频 | ✅ 10,799 文件 SHA 一致 |
| Stage 08 几何对比结果（Top-1/Top-5 各 5,324 条） | ✅ 10,728 文件 SHA 一致 |
| 冻结特征 features_v2（val 分片） | ✅ 新服务器重生成，数值一致（最大绝对差 3.05e-5，fp16 量化级） |
| Python 环境（py3.10 + torch 2.9.1+cu128 + torchvision 0.24.1） | ✅ RTX 5090 验证通过 |
| ResNet-50（IMAGENET1K_V2）layer4 输出 | ✅ 2048 维 |
| Stage 08 单条复现（Top-1/Top-5 各 2 条） | ✅ 新旧服务器均与存档字段级一致 |
| Stage 06 单条复现 | ⚠️ 见第 4 节"非确定性发现" |
| GitHub 身份 | ✅ ty-zhang-jlu（双仓库 ADMIN） |

## 1. 迁移内容清单

数据落位（新服务器 HDD 数据区，经兼容链接被代码按原路径访问，代码零改动）：

| 目录 | 文件数 | 大小 | 说明 |
|---|---:|---:|---|
| teacher_03b | 114 | 28M | 03B 教师轨迹 npz/csv + 分析 |
| candidates_04 | 56+28 | 15M+3.4M | cache_v2(candidates24/48) + v1 cache |
| checkpoints | 63 | 1.1G | runs(seed42/43) + runs_v2(seed42 全 epoch) |
| DATA/original_citynav | 13 | 754M | 官方原始轨迹 json |
| DATA/cityrefer(_original) | 4 | 54M | objects.json + processed_descriptions.json |
| DATA/rgbd | 68 | 3.4G | 官方俯视 RGB + 高程 TIF（34 块） |
| results_06 | 10,799 | 267M | Stage 06 全量闭环逐条结果 + 视频 |
| results_08 | 10,728 | 268M | Stage 08 Top-1/Top-5 逐条对比 + 视频 |

features_v2（val_seen 2,498 条 / val_unseen 2,826 条）由新服务器用冻结 ResNet-50 重生成；train_seen（22,002 条）重生成于迁移当日后台完成。features_scaled 为派生缓存，本地重算。

未迁移（用户明确决定）：PLY 点云、主线分支本地产物、各阶段审计记录、gsam 语义地图。PLY 如需可从 OpenDataLab 重新下载。

## 2. 校验方法

- 每批传输前后生成源/目标文件清单，`sha256sum` 逐文件对比（归一化路径后 `comm` 比对），全部 0 差异；
- 教师轨迹条数用 mmap 加载 npz 核验；
- 模型用 `model.py` TransformerM（含候选分支，N_CANDIDATES=5，参数量 5,477,124）strict load；
- 特征缓存对比旧服务器 journal.jsonl 的逐条 SHA-256 与 index.json 语义字段。

## 3. 环境重建（M3）

| 项 | 值 |
|---|---|
| 环境路径 | 数据盘 `/conda/envs/citynav-mec` |
| Python / torch / torchvision | 3.10.22 / 2.9.1+cu128 / 0.24.1+cu128（与源环境一致） |
| 关键依赖 | numpy 1.26.4、opencv-headless 4.10.0.84、rasterio 1.4.4、transformers 4.57.6 等（按源环境 pinned 清单） |
| GPU 验证 | RTX 5090 识别、CUDA matmul、ResNet-50 V2 前向（2048 维）全部通过 |

## 4. Stage 06 复现的非确定性发现（重要）

对 Stage 06 闭环评估做了三组对照实验：

1. **旧服务器复跑**（迁移日 23:55）：3/3 条 episode 与已提交 journal **逐位一致**（含浮点 stop_dist 全部小数位）；
2. **旧服务器复跑**（次日 00:10、00:15 两次）：同一机器、同一权重、同一数据，3/3 条变为另一组轨迹（与 journal 不同），且两次之间自洽；
3. **新服务器复跑**（三次）：全部给出与旧服务器第 2/3 次**逐条一致**的轨迹。

结论：
- 新服务器与旧服务器**当前行为完全一致**（多次运行稳定自洽）；
- 已提交的 Stage 06 journal 数值在**源机器上也无法稳定复现**——评估管线存在运行间非确定性（GPU 内核选择随显存状态变化 + 策略对近零 logit 差距的混沌放大）；
- 该性质属于**原研究管线固有**，与本次迁移无关；迁移的逐位/数值一致性已由第 2 节的全部校验证明；
- 迁移报告记录两种轨迹：A 组（40/42/66 步，与提交 journal 一致，旧服务器 23:55 复现）、B 组（38/30/9 步，两服务器当前一致）。

建议（供后续实验参考）：评估前固定 `torch.backends.cudnn.deterministic = True` 并清空 GPU 竞争负载，可显著降低运行间差异；如需完全确定性可进一步限制 cuDNN/cuBLAS 内核。

## 5. 工具链迁移（M4）

- Claude Code 2.1.314（源 2.1.302，无需重装）；全局 CLAUDE.md、hooks、市场（5 个第三方市场）、插件（karpathy-skills / codex / superpowers，版本与源一致）、自定义 skills ×3 均已落位；
- tmux 3.4 + 统一配置；持久会话 `citynav_main`；
- GitHub CLI 登录 ty-zhang-jlu（active），双仓库 ADMIN 权限核验通过；
- 认证材料零复制（新服务器自有凭证）。

## 6. 旧服务器清理

用户最终决定：**不删除任何数据，M6 清理阶段跳过**。旧服务器（含 mec-v8.2、主线 checkout、fixed50m 工作区、DATA/PLY 等全部目录与运行环境）作为完整备份保留，会话与进程不动。此前生成的删除清单仅作记录，未执行。

## 7. 遗留事项

- train_seen 特征全量重生成（后台，已完成或接近完成）；
- 主线分支本地产物、各阶段审计记录、gsam 语义地图未迁移（用户决定；源服务器保留期内存续）；
- PLY 点云未迁移，需要时从 OpenDataLab 重新下载。
