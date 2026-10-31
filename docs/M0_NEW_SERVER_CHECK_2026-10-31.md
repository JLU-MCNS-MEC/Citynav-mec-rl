# M0 新服务器现场检查与同步报告

- 日期：2026-10-31
- 服务器：新服务器（hostname `ubuntu`，用户 `tenant2`）
- 分支：`exp/citynav-mec-fixed50m-topdown`
- 目标：确认新服务器可承接 Stage 10（10A–10E）全部实验

## 1. 机器与环境

| 项目 | 结果 |
|---|---|
| hostname / 用户 | `ubuntu` / `tenant2`（uid 1004，render/video 组 ✓） |
| 系统 | Ubuntu 24.04.4 LTS，内核 6.17.0-35-generic |
| GPU | **NVIDIA GeForce RTX 5090，32 GB**（32607 MiB），空闲 0% |
| 驱动 / CUDA | 580.159.03 / 13.0（torch 用 cu128 后端，已验证 cuda available） |
| CPU | Intel i9-14900KF，24 核 32 线程 |
| 内存 | 62 GiB（当前可用约 26 GiB） |
| 磁盘 | `/` 剩 123 G（86% 已用，注意）；`/home/tenant2/dataext` 剩 43 G；`/mnt/hdd2t` 剩 1.8 T；**`/mnt/hdd8t` 剩 7.1 T（数据/大产物主盘）** |
| Python（base） | miniconda3，Python 3.14.6 |
| Conda 环境 | `base`（miniconda3）；**`citynav-mec`（/mnt/hdd2t/conda/envs/citynav-mec）** |
| 实验环境可用性 | ✓ 可用：Python 3.10.22、torch 2.9.1+cu128、torchvision 0.24.1+cu128、rasterio 1.4.4、opencv 4.10.0、numpy 1.26.4、pandas 2.3.3、matplotlib、scipy、pyyaml、tqdm |
| 缺失依赖 | `scikit-image`、`imageio` 未安装（仅影响部分分析/视频脚本；10A 开始前 `pip install scikit-image imageio imageio-ffmpeg` 即可，约 2 分钟） |

## 2. tmux / Git / GitHub 身份

- tmux 3.4 已安装 ✓；本会话运行于 tmux 会话 `citynav_main`（attached）✓
- 遗留会话：`citynav_migration`（旧服务器迁移任务，已完成、空闲，未运行任何进程）；`work-0`（其他工作会话，与 CityNav 无关，未触碰）
- 无遗留训练/评测进程：仅一个 PyCharm 交互式 python 控制台（citynav-mec 内核，空闲）
- git 身份：`user.name = ty-zhang-jlu`，`user.email = 253254052+ty-zhang-jlu@users.noreply.github.com` ✓
- `gh auth`：active account = **ty-zhang-jlu** ✓（HTTPS 协议，走 mihomo 代理，push/pull 已验证可用）
- 关联仓库 `JLU-MCNS-MEC/mec-v8.2`：gh 可访问（main 分支），保持可关联 ✓

## 3. 本地代码目录

| 目录 | 说明 |
|---|---|
| `/mnt/hdd2t/workspaces/Citynav-mec-rl` | 主克隆，分支 `main`，HEAD `01837c0`，与 origin/main 一致，工作区干净 |
| `/mnt/hdd2t/workspaces/Citynav-mec-rl-fixed50m` | **实验克隆（Stage 10 在此进行）**，分支 `exp/citynav-mec-fixed50m-topdown`，已同步到 `a2bf270` |
| `/mnt/hdd8t/Citynav-MEC` | 数据/大产物目录（rgbd、checkpoints、results_06/08、teacher_03b、candidates_04） |

本地工作区状态（保留、不视为错误）：
- `experiments/04_.../pipeline.py`：本次 M0 打的最小路径适配补丁（见 §5，后续随 Stage 10 提交）
- `experiments/05_transformer_m_training/results/normalization_check.json`：迁移时路径修正（旧路径 → `/mnt/hdd2t/...`），属结果文件本地修正，不入库
- 未跟踪目录均为到 `/mnt/hdd8t/Citynav-MEC` 的软链或本地缓存（runs、runs_v2、cache、cache_v2、results_naive_15deg），为迁移产物，保留

## 4. GitHub 同步结果

旧服务器推送的 5 个新提交已全部同步到新服务器本地（fast-forward，无冲突）：

| commit | 内容 |
|---|---|
| `237a55b` | Stage 09A：在线目标匹配（冻结 ResNet layer2 中位 3.6 m） |
| `9312f45` | Stage 09B：动态信念闭环 SR 47.6% |
| `aa0bdd9` | Status：记录 stage 09 commit SHA |
| `a1296e8` | Stage 09C：连续定位 C1 SR 64.7%；硬门控 C2 61.2% |
| `a2bf270` | Status：记录 stage 09C commit SHA（当前 HEAD） |

- `experiments/09_online_visual_goal_matching/09C_continuous_localization/` ✓ 完整：`09C1_REPORT.md`、`09C2_REPORT.md`、`REPORT.md`、`CONFIG.yaml`、scripts（run_09c / continuous_env / continuous_belief / calibrate / analyze / videos）、results（09c_c1 + 09c_c2 各 5,324 行 journal）、figures、videos（3 个 MP4）
- `experiments/STATUS.md` ✓ 已是最新（2026-10-31 10:30 更新，含 09C 行）
- `experiments/10_goal_query_mec_training/` 不存在 → 按计划在 Stage 10A 开始前创建
- 主仓库 `main` 与 origin/main 一致 ✓；`mec-v8.2` 可关联 ✓

## 5. 数据与依赖核查

| 依赖 | 位置 | 状态 |
|---|---|---|
| 原始俯视 PLY 地图 | `/mnt/hdd8t/Workspace/DATA/SensatUrban_Dataset/ply/train/`（birmingham/cambridge 各 block） | ✓ |
| 俯视 RGB 栅格（PNG+TIF，48 MB/图） | `/mnt/hdd8t/Citynav-MEC/DATA/rgbd/`（3.4 G） | ✓ |
| 任务拆分 JSON | `/mnt/hdd8t/Citynav-MEC/DATA/original_citynav/`（train_seen / val_seen / val_unseen / test_unseen 及 easy/medium/hard 细分） | ✓ |
| 03B 教师轨迹 | `experiments/03b_target_directed_teacher/results/teacher_{train_seen,val_seen,val_unseen}.{csv,npz}` + obstacles 缓存（每 block .json/.npz） | ✓ |
| 候选库 | `experiments/04_.../cache_v2/candidates24`（软链 → hdd8t）、candidates48 | ✓ |
| Stage 06 评测清单 | `experiments/06_closed_loop_success_rate/results/seed42/`（manifest.json + manifest.sha256，5,324 任务） | ✓ |
| Stage 08 Top-1 journal | `experiments/08_.../results/top1/journal.jsonl` | ✓ |
| ResNet-50 冻结骨干权重 | torchvision `ResNet50_Weights.IMAGENET1K_V2`，已缓存 `~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth`（102 MB） | ✓ |
| 09C-1 代码可运行性 | **冒烟测试通过**：c1 模式 2 条任务 0.82 ep/s（RTX 5090），journal/断点续跑机制正常（冒烟产物已清理） | ✓ |
| 迁移一致性 | `logs/verify_all.log`：03b / 04 cache_v2 / 05 runs / 06 / 08 全部 SHA-256 与旧服务器一致 | ✓ |

### 关键适配（已做）

1. `pipeline.py` 的 `DATA = /home/20261024_1/Workspace/DATA`（旧服务器绝对路径）改为
   `DATA = Path(os.environ.get("CITYNAV_DATA", "/home/20261024_1/Workspace/DATA"))`。
   新服务器运行任何需要 rgbd 的脚本时设置 `CITYNAV_DATA=/mnt/hdd8t/Citynav-MEC/DATA`。
   默认值保留旧路径，历史复现不受影响。
2. `09C/scripts/run_full.sh` 的 `PY=/home/20261024_1/miniconda3/envs/DRL_migrated/bin/python`
   为旧服务器路径；Stage 10 起统一使用 `/mnt/hdd2t/conda/envs/citynav-mec/bin/python`。
3. `manifest.json` 中 `candidate_library` 字段仍为旧服务器路径，但代码实际按仓库相对路径
   解析候选库（`dataset_v2.py` 的 `CANDS`），不受影响；09C 冒烟已验证。

## 6. 结论

- **机器可用**：RTX 5090 32 GB 空闲，实验环境完整，断点续跑与 tmux 就绪。
- **代码已同步**：`exp/citynav-mec-fixed50m-topdown` 已到 GitHub 最新 `a2bf270`（09C 完成态）。
- **数据完整**：09C-1 连续目标定位在新服务器实测可运行（冒烟 2/2 通过）。
- **可以继续 Stage 10** ✓（唯一待办：安装 skimage/imageio；创建 `experiments/10_goal_query_mec_training/` 目录结构）。

## 7. Stage 10 各阶段预计耗时（基于 09C 冒烟 0.82 ep/s 与 Stage 05 训练 80 min 校准）

| 阶段 | 内容 | 预计耗时 |
|---|---|---|
| 10A | PQ / CA 匹配模块训练（Train-Seen）+ 离线定位评估（5,324×2 拆分） | 3–5 小时 |
| 10B | C1 / PQ / CA 三种方案 × 5,324 条闭环（单进程 ~1.8 h/轮 × 3 + 统计） | 6–9 小时 |
| 10C | MEC 输入契约 + cross-fitting 特征生成 + 审计 | 6–8 小时 |
| 10D | Transformer-M 正式训练（30 epoch，seed 42） | 2–4 小时 |
| 10E | 最终闭环验证（5 种控制器对比，5,324 条）+ 视频 | 3–5 小时 |
| 合计 | | 约 20–31 小时（含报告撰写与推送） |

## 8. 说明与承诺

- 09C-1（C1）保留为正式基线；C2 不再使用、不再调试，仅作负结果归档。
- Stage 10 所有新产物写入 `experiments/10_goal_query_mec_training/`，不修改任何历史结果目录。
- 全程 tmux（本会话已在 `citynav_main` 内），全部长任务断点可续（journal / checkpoint 原子保存）。
- 训练前、中、后按 10 分钟或子阶段粒度中文汇报耗时与指标。
