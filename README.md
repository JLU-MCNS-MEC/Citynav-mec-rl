# CityNav-MEC-RL

CityNav / CityRefer 数据适配、视觉观测适配，以及 MEC-V8.2 Small / Transformer-M
模型的最小前向与单步训练验证。

当前阶段**只做数据与模型接口的 Smoke Test**：不启动大规模训练，不实现完整
RL 环境，不采用 CityNav 论文的 GSM 语义地图增强。

## 当前范围

已完成：

- 读取 CityNav 官方 `original_citynav/citynav_*.json` 与 CityRefer `objects.json`；
- 校验 `area / block / object_ids / ann_ids / descriptions / trajectory /
  marker_positions / target_positions / split` 字段；
- 将 CityNav 六元组 `(x, y, z, dx, dy, dz)` 转成 `(x, y, z, yaw, pitch)`，
  并对近垂直方向做确定性的 yaw 回退；
- 按固定离地高度生成 5-D 位姿，保留原始轨迹用于审计；
- 复现官方 CityNav top-down 观测裁剪（正射 RGB + DSM 高度代理）；
- 生成 3 个尺度的目标参考裁剪和 8×8 视觉余弦热图；
- 运行冻结视觉编码器、冻结文本编码器、Small 与 Transformer-M 前向；
- 运行一次监督训练步（movement CE + arrival BCE），检查 loss、梯度和参数更新；
- 对共享原始数据做运行前后元数据快照，证明未被修改。

尚未实现：

- CityNav GSM 语义地图或论文中的导航增强结构；
- 完整 RL 环境、reward、rollout 与策略训练；
- 真正的 PLY 第一人称 RGB/depth 渲染（当前仅审计 PLY 头和少量顶点）；
- 20-MEC 网络状态的真实测量（Smoke Test 使用确定性占位向量）；
- 完整训练配方与验证集指标。

## 与 CityNav 官方、MEC-V8.2 的关系

- **观测几何**：`TopDownObservationAdapter._view_area_corners` 与官方
  `vlnce/space.py:view_area_corners` 一致：裁剪区域的半边长等于离地高度，
  因此边长为 `2 × altitude`。默认离地高度 50 m。
- **Depth 通道**：沿用官方 CityNav 的 `pose.z - height` 约定，是“离地高度”
  代理，不是视线方向深度。
- **动作接口**：保留 MEC-V8.2 的 4 动作接口 `Forward / Left / Right / Stop`，
  参数为 5 m 前进、15° 转向、20 m 成功半径。CityNav 官方离散动作使用 ±30°
  转向；这里刻意没有改成官方参数，以保持 MEC 策略接口不变。
- **模型结构**：`SmallMEC` 对应 V8.2 `MeanPoolMLP`，`TransformerMMEC` 对应
  `ExplicitReasoningTransformer`；唯一新增是冻结语言特征上的小型线性投影和
  一个语言 token。热图由当前/目标视觉 token 的余弦相似度生成，不含 GSM、
  模拟器掩码、绝对坐标或真值距离。

## 目录结构

```
configs/                 # 可提交的配置模板；本地绝对路径配置被忽略
data_cache/              # 本地缓存（忽略）
docs/                    # 审计 / Smoke Test 报告
figures/                 # 必要结果图片
reports/                 # 运行生成的报告（忽略）
runs/                    # 训练产物（忽略）
scripts/
  audit_dataset.py       # 只读数据审计
  smoke_test.py          # CPU 优先的端到端 Smoke Test
src/
  config.py              # 配置对象与环境变量展开
  data/
    citynav_dataset.py   # episode / split / CityRefer 读取
    trajectory_adapter.py# 六元组位姿与纯追踪教师
    observation_adapter.py # top-down 观测、目标裁剪、PLY/栅格审计
    language_adapter.py  # 冻结 BERT 或离线 hashing 文本特征
    paths.py             # 写入隔离与原始数据快照
  models/                # 视觉编码器、Small、Transformer-M
  training/supervised.py # 监督损失与单步训练
tests/                   # 单元 / 契约 / 真实数据集成测试
```

## 环境

要求 Python ≥ 3.9。本机验证环境为 Python 3.10、torch 2.9.1+cu128，Smoke Test
默认在 CPU 上运行。

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 不锁定 CUDA 版本；请按服务器的 GPU 驱动安装匹配的
PyTorch。不要在有其他 GPU 任务运行时默认把 Smoke Test 放到 GPU 上。

## 数据配置

原始 CityNav / CityRefer / RGBD / PLY 目录**只读**使用。仓库不包含、也不复制
任何原始数据。数据路径只通过本地的、被 `.gitignore` 排除的配置文件或环境变量
提供：

```bash
cp configs/data_config.example.yaml configs/data_config.local.yaml
# 编辑 data_root / ply_root，或：
export CITYNAV_DATA_ROOT=/path/to/hett-inputs/data
export CITYNAV_PLY_ROOT=/path/to/original_block_ply
```

`configs/data_config.example.yaml` 只使用 `${CITYNAV_DATA_ROOT}` 和
`${CITYNAV_PLY_ROOT}` 占位符，不包含服务器绝对路径。

已知数据事实（由 `scripts/audit_dataset.py` 验证）：

- CityRefer：34 个 city block，26,274 个标注对象；
- 三个开发划分覆盖 28 个 block；`test_unseen` 另有 6 个 block；
- 划分计数：train_seen 22,002 / val_seen 2,498 / val_unseen 2,826 /
  test_unseen 5,311；
- `ann_ids` 是对象 `descriptions` 列表的 0-based 索引；
- CityNav split 描述与 CityRefer 对象描述存在少量**空白和 ASCII 双引号**
  差异（例如 `"L"` vs `L`）；严格匹配约 97.4%，归一化空白和双引号后
  三个开发划分均为 100%；
- 目标位置取 `target_positions[-1]`，与 CityRefer 对象 `position` 对齐；
- RGB 与 DSM 栅格的像素尺寸随 block 裁剪范围变化（不全是 4000×4000），
  但每个 block 的 PNG/TIF 尺寸一致，地理变换均为 0.1 m。

## 快速开始

```bash
# 1. 单元与契约测试（真实数据集成测试在数据不可用时会自动 skip）
pytest -q

# 2. 只读数据审计（默认不包含 test_unseen）
python scripts/audit_dataset.py --splits val_seen val_unseen

# 3. 端到端 Smoke Test（CPU，默认 2 个 val_unseen episode）
python scripts/smoke_test.py --split val_unseen --episodes 2 --device cpu

# 可选：使用本地 HuggingFace 缓存中的 bert-base-uncased
python scripts/smoke_test.py --text-backend bert
```

`test_unseen` 默认被数据集和审计脚本拒绝；只有显式传
`--allow-test-unseen` / `--include-test-unseen` 才能读取，保留给方案冻结后的
最终报告。

## 输出与数据安全

- `reports/`、`runs/`、`data_cache/`、`logs/`、TensorBoard event、checkpoint
  和 `configs/*.local.yaml` 均被 `.gitignore` 排除；
- `reports/` 与 `runs/` 保留 `.gitkeep`，目录结构可提交；
- `scripts/smoke_test.py` 在运行前后调用 `snapshot_raw_dirs()`，如果共享原始
  数据的文件数、大小、mtime 或摘要变化会直接失败；
- 所有写操作都经过 `src/data/paths.py:ensure_output_dir`，拒绝写到项目目录
  之外或原始数据目录内。

## 测试覆盖

- `tests/test_trajectory_adapter.py`：六元组转换、yaw 回退、固定高度、
  4 动作转移、纯追踪教师成功/失败；
- `tests/test_dataset_contract.py`：字段契约、描述匹配、split 字段一致性、
  `test_unseen` 守卫、最小 CityRefer 树读取；
- `tests/test_models_and_training.py`：Small / Transformer-M 前向、联合分布
  归一化、损失、单步参数更新；
- `tests/test_observation_adapters.py`：官方裁剪几何、PLY 头解析、栅格到张量；
- `tests/test_config_paths.py`：环境变量展开、路径脱敏、输出目录隔离；
- `tests/test_repository_layout.py`：Git 根目录必须精确指向本项目；
- `tests/test_real_data_integration.py`：真实 val_unseen episode 的观测、编码器
  与 Small 前向（无本地数据时 skip）。

## 下一步

1. 在 train_seen 上跑完整的监督训练基线，并在 val_seen / val_unseen 上评测
   SR / SPL / NE；
2. 将 MEC-V8.2 的完整训练配方（movement class balance、arrival hard states、
   checkpoint 选择）移植到本仓库；
3. 在真实测量或模拟的网络状态上替换占位 5-D 向量；
4. 再决定是否接入 RL 环境与策略训练，以及是否需要 PLY 第一人称渲染。

## Git

```bash
git remote -v
# origin  https://github.com/JLU-MCNS-MEC/Citynav-mec-rl.git
```

远程默认分支为 `main`。本地提交前请确认：

```bash
git rev-parse --show-toplevel       # 必须是本仓库根目录
git status --short                  # 不应出现 ../ 或原始数据路径
```

不要提交 `configs/*.local.yaml`、原始数据、checkpoint 或大型生成文件。
