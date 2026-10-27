# CityNav-MEC-RL

CityNav / CityRefer 数据适配、SensatUrban PLY 第一视角 RGB/Depth 渲染、官方
六动作监督，以及 MEC-V8.2 Small / Transformer-M 的最小训练验证。

当前阶段**只做数据、观测、六动作标签、T5+语言候选检索与过拟合/Smoke 验证**：
不启动完整训练集训练，不实现完整 RL 闭环，不采用 CityNav 论文的 GSM 语义地图增强。

## 当前协议

- 视觉输入：原始 SensatUrban PLY 的第一视角 RGB + 透视前向 Depth；
- 文本输入：冻结预训练 `bert-base-uncased`；正式训练不使用 hashing；
- 视觉编码器：MEC-V8.2 `shared_encoder.pt` 初始化，本阶段允许更新；
- 模型：Small 与 Transformer-M 主干保留，移动头 5 类 + 二元 Stop/Arrival 头；
- 动作：官方 CityNav 六类离散动作；
- 教师：官方 lookahead 教师，对 marker 5 m 停止；评估成功半径单独为 20 m；
- 数据：只读官方 `original_citynav` 四个划分；`test_unseen` 默认封存。

## 六动作

| ID | 动作 | 执行 |
|---:|---|---|
| 0 | STOP | 不移动 |
| 1 | MOVE_FORWARD | 前进 5 m |
| 2 | TURN_RIGHT | 右转 30° |
| 3 | TURN_LEFT | 左转 30° |
| 4 | GO_UP | 上升 2 m |
| 5 | GO_DOWN | 下降 2 m |

移动头输出 5 类，顺序为 Forward / Right / Left / Up / Down；Stop 来自 Arrival
头。`combine_move_and_arrival` 组合成 `[Stop, Forward, Right, Left, Up, Down]`
的 6 类联合分布，与官方动作 ID 对齐。

## T5 + 语言检索

ToyEnv V8.2 T5 六模态 + 冻结 BERT：

| 模态 | 维度 |
|---|---|
| Current RGB tokens | 576 × 384 |
| Goal RGB tokens（Top-3 检索候选） | 3 × 384 |
| History visual | 16 × 768 |
| Pose history | 16 × 12（6 连续 + 6 动作 one-hot） |
| Action history | 16 × 6 |
| History mask | 16 |
| GAB | 20 |
| Top-5 map retrieval | 22 |
| Language（冻结 BERT） | 768 |

候选库只从官方正射图按固定网格（stride 48 m）和多尺度（32/64/128 m）
生成；**不使用 CityRefer 目标框、目标 ID、`object_ids`、`ann_ids` 或目标裁剪**。
训练阶段只用 train_seen 的 `target_positions` 给网格候选取正负标签；推理端
`retrieve_candidates` 只接收语言特征、候选视觉特征和 Top-K，不接受目标坐标、
ID、裁剪、教师动作或成功标签。

- full：Top-3 候选视觉 token 作为 Goal RGB，Top-5 用于 22-D map 特征；
- map-free baseline：去掉 Goal/GAB 地图证据和 Top-5 map 分支，使用
  current/history/pose/language；
- 检索不足时 Goal/Map 零填充并记录 `goal_valid`；不回落使用真实目标。

本次小规模结果：val_seen Recall@1/5/10 = 0.348/0.652/0.739，val_unseen =
0.021/0.146/0.292；full 相对 map-free **没有显示正增益**。这是必须保留的负结果，
详见 [`docs/T5_RETRIEVAL_REPORT.md`](docs/T5_RETRIEVAL_REPORT.md)。

## 第一视角渲染

`src/data/fpv_renderer.py` + `src/data/fpv_observation.py`：

- 原始 PLY 以 `np.memmap` 只读打开，按半径裁剪，不复制原始数据；
- 针孔相机：方形图、90° HFOV、`fx=fy=W/2`、主点 `(W-1)/2`；
- 坐标系：CityNav / SensatUrban 局部 `x/y` 水平、`z` 向上；XY/Z 恒等变换；
- yaw = `atan2(dy, dx)`，pitch = `atan2(dz, hypot(dx, dy))`；正式路径使用轨迹
  方向向量的 pitch；
- Depth = 相机前向深度 `dot(point-camera, forward)`，单位 m；不是欧氏距离，
  也不是俯视高度图；
- 逐像素最小深度 z-buffer；点云渲染没有表面重建，空洞是预期现象；
- 不是 AirSim 渲染，不声称与官方 AirSim 帧完全一致。

详细相机参数、样例指标、空洞与坐标校验见
[`docs/FPV_RENDER_REPORT.md`](docs/FPV_RENDER_REPORT.md)。

## 模型与文本

- `SmallMEC`：V8.2 `MeanPoolMLP` 的五类移动头 + Stop 头；
- `TransformerMMEC`：V8.2 `ExplicitReasoningTransformer` 的五类移动头 + Stop 头；
- 语言：冻结 BERT 池化特征（768 维）经小型投影层进入 MEC；
- 视觉：`VisualTokenEncoder`（336×336，24×24 patch，576×384 tokens）；
- 热图：当前/目标视觉 token 余弦相似度池化到 8×8；不含 GSM 或真值距离。

## 数据事实

- CityRefer：34 个 block、26,274 个标注对象；
- 三个开发划分覆盖 28 个 block；`test_unseen` 另有 6 个 block；
- 官方计数：train_seen 22,002 / val_seen 2,498 / val_unseen 2,826 /
  test_unseen 5,311；
- `ann_ids` 是对象 `descriptions` 的 0-based 索引；
- split 文本与 CityRefer 文本存在空白/ASCII 双引号差异；严格匹配约 97.4%，
  归一化后三个开发划分均为 100%；
- `target_positions[-1]` 是评估目标；`marker_positions[-1]` 是教师停止目标；
  两者不能混用；
- 原始轨迹高度保留，不再固定 50 m；
- PNG/TIF 尺寸随 block 裁剪变化，但同一 block 的 PNG/TIF 尺寸一致，变换均为
  0.1 m。

## 目录

```
configs/                 # 可提交配置模板；本地绝对路径配置被忽略
data_cache/              # 缓存、编码器权重、过拟合 checkpoint（忽略）
docs/                    # FPV / 过拟合 / Smoke 报告
figures/fpv_examples/    # 真实 FPV RGB/Depth/overlay 示例图
reports/                 # 运行生成的报告（忽略）
runs/                    # 训练产物（忽略）
scripts/
  audit_dataset.py       # 只读数据审计
  render_fpv_examples.py # 少量真实 FPV 样例 + 方向校验
  overfit_fpv_smoke.py   # 128–256 轨迹六动作过拟合验证
  t5_smoke.py            # T5+语言候选检索/Token/短训练 Smoke
  smoke_test.py          # 单 episode 前向/单步训练 Smoke
src/
  data/
    actions.py           # 官方六动作
    citynav_dataset.py   # 数据/描述/目标读取
    trajectory_adapter.py# 六元组位姿 + 官方 lookahead 六动作教师
    observation_adapter.py # top-down 诊断、目标裁剪、PLY memmap/裁剪
    fpv_renderer.py      # CPU 点云针孔渲染与深度
    fpv_observation.py   # 第一视角观测适配器
    candidates.py        # 固定网格/多尺度正射候选库、Top-K/Recall
    gab.py               # 8-bin 因果 GAB
    t5_features.py       # 16 步历史/位姿/动作/语言维度
    language_adapter.py  # 冻结 BERT（含离线 hashing 测试回退）
    paths.py             # 写入隔离与原始数据快照
  models/                # 视觉编码器、Small、Transformer-M、T5 MEC
  training/
    supervised.py        # 因子化六动作损失与单步训练
    retrieval.py         # 语言→地图候选检索器
tests/                   # 单元/契约/防泄漏/真实数据测试
```

## 环境

要求 Python ≥ 3.9。本机验证环境：Python 3.10、torch 2.9.1+cu128；渲染默认
CPU，训练验证可用 GPU。

```bash
python -m pip install -r requirements.txt
```

有 GPU 任务运行时不要并发启动 GPU 训练；本仓库的 Smoke Test 默认 CPU。

## 数据配置

原始 CityNav / CityRefer / RGBD / PLY 只读使用，仓库不包含也不复制原始数据。

```bash
cp configs/data_config.example.yaml configs/data_config.local.yaml
# 编辑 data_root / ply_root，或：
export CITYNAV_DATA_ROOT=/path/to/hett-inputs/data
export CITYNAV_PLY_ROOT=/path/to/original_block_ply
```

`configs/data_config.local.yaml` 被 `.gitignore` 排除。

## 编码器权重与 BERT

- BERT：使用本地 HuggingFace 缓存中的 `bert-base-uncased`；不提交权重。
- 视觉编码器：把 MEC-V8.2 的 `shared_encoder.pt` 放到
  `data_cache/weights/shared_encoder.pt`（本地忽略），或用
  `--encoder-checkpoint` 指定；不提交权重。
- `configs/train_config.example.yaml` 的 `encoder.checkpoint` 可填本地路径。

## 快速开始

```bash
# 1. 测试
pytest -q

# 2. 数据审计（默认不含 test_unseen）
python scripts/audit_dataset.py --splits train_seen val_seen val_unseen

# 3. 少量真实 FPV 样例（512，含 Depth/overlay/俯视校验）
python scripts/render_fpv_examples.py \
  --splits val_seen val_unseen --samples 4 --scan-records 5000 \
  --render-size 512 --crop-radius 200 --splat-radius 1

# 4. 128 轨迹六动作过拟合验证（BERT 冻结；视觉编码器 + MEC 更新）
python scripts/overfit_fpv_smoke.py \
  --trajectories 128 --frames-per-trajectory 2 \
  --steps 300 --batch-size 8 \
  --models small transformer_m \
  --device cuda \
  --render-size 336 --crop-radius 120 \
  --max-points 4000000 --splat-radius 0

# 5. T5 + 冻结 BERT 候选检索/Token/短训练 Smoke
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
  --render-size 336 --crop-radius 120 \
  --candidate-stride 48 --candidate-scales 32 64 128

# 6. 单 episode FPV Smoke（hashing 仅用于快速离线检查）
python scripts/smoke_test.py \
  --split val_unseen --episodes 1 --observation fpv \
  --render-size 336 --crop-radius 120 --device cpu --text-backend hashing
```

`test_unseen` 默认被数据集和审计脚本拒绝；只有显式
`--allow-test-unseen` / `--include-test-unseen` 才能读取，保留给最终冻结报告。

## 已验证结果

- `pytest -q`：72 passed；
- FPV 样例：4 个不同地图的 512×512 RGB/Depth/overlay，见
  `figures/fpv_examples/`；
- 方向校验：pitch=-90 点云 vs 官方正射，yaw=0 相关最高（birmingham_block_11：
  0.610 vs 90/180/270 的 0.024/0.186/0.027）；
- 128 轨迹、256 帧、300 步过拟合：
  - Small：六类准确率 0.125 → 0.613，Stop P/R 0.857/0.750；
  - Transformer-M：0.125 → 0.598，Stop P/R 0.742/0.719；
  - BERT 梯度 0、参数不变；视觉编码器与 MEC 参数均更新；
  - 峰值显存约 1.31–1.44 GB（RTX 5090，batch=8）；
- 原始数据元数据摘要运行前后一致；
- T5+语言 Smoke：12 个地图、1,602 个候选、456 帧；val_seen R@1/5/10 =
  0.348/0.652/0.739，val_unseen = 0.021/0.146/0.292；full 相对 map-free
  **未显示正增益**（负结果，不能声称地图辅助提升）。

详细结果：
[`docs/FPV_RENDER_REPORT.md`](docs/FPV_RENDER_REPORT.md)、
[`docs/OVERFIT_REPORT.md`](docs/OVERFIT_REPORT.md)、
[`docs/T5_RETRIEVAL_REPORT.md`](docs/T5_RETRIEVAL_REPORT.md)、
[`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md)、
[`docs/SMOKE_TEST_REPORT.md`](docs/SMOKE_TEST_REPORT.md)。

## 输出与数据安全

- `reports/`、`runs/`、`data_cache/`、`logs/`、TensorBoard event、checkpoint
  和 `configs/*.local.yaml` 均被 `.gitignore` 排除；
- `reports/`、`runs/`、`data_cache/` 保留 `.gitkeep`；
- 所有写操作经过 `src/data/paths.py:ensure_output_dir`，拒绝写到项目外或原始
  数据目录；
- `scripts/overfit_fpv_smoke.py` 默认在运行前后调用 `snapshot_raw_dirs()`，原始
  数据摘要变化会直接失败；
- 不提交原始 JSON / CityRefer / RGB/TIFF / PLY / 大缓存 / 权重 / 日志。

## 限制与下一步

限制：

- 点云 FPV 有空洞、无表面重建/光照，不等于 AirSim；
- 部分地图的点云与正射颜色相关性较低，正式实验前需逐地图几何配准；
- 过拟合验证只使用 128 条 train_seen 轨迹和 256 帧，不代表 val 泛化；
- T5 候选检索在 val_unseen 上很弱（R@1 0.021），当前不足以支持地图增益；
- 网络状态为确定性 5 维占位向量；
- 尚未移植 V8.2 完整训练配方（movement class balance、arrival hard states、
  checkpoint 选择）；
- 尚未做完整 SR/SPL/NE 评测、RL、DQN/PPO 或 GSM。

下一步：

1. 改进无泄漏候选检索：更多 train episodes/训练步、更强的跨模态表示和候选覆盖，重新评估 R@1/5/10；
2. 逐地图检查 PLY/轨迹/目标/语言/FPV 对齐，排除低相关地图；
3. 在完整 `train_seen` 上跑监督基线，在 `val_seen` / `val_unseen` 评测；
4. 移植 V8.2 训练配方与真实网络状态；
5. 再决定 RL 闭环与全量 FPV 缓存。

## Git

```bash
git remote -v
# origin  https://github.com/JLU-MCNS-MEC/Citynav-mec-rl.git
```

远程默认分支 `main`。提交前确认：

```bash
git rev-parse --show-toplevel       # 必须是本仓库根目录
git status --short                  # 不应出现 ../ 或原始数据路径
```

不要提交本地配置、原始数据、缓存、checkpoint 或大型日志。
