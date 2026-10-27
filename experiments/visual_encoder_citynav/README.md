# Visual Encoder CityNav

独立实验目录：CityNav 共享视觉编码器预训练。

- 复用仓库现有 `VisualTokenEncoder`、FPV 渲染器、CityNav 数据适配器、动作定义和候选库；
- 不训练 Small/Transformer-M、不生成正式 MEC Token、不开发 RL；
- 最终只保存视觉编码器；动作/跨视角/检索辅助头不进入最终模型。

完整协议见 [`PROTOCOL.md`](PROTOCOL.md)。

## 文件

```
configs/pretrain.yaml       # 训练配置模板
data.py                     # point-cloud cache / manifest / dataset / candidate masks
model.py                    # encoder + 临时辅助头；严格加载 shared_encoder.pt
losses.py                   # 多正样本 InfoNCE + 动作 CE
train.py                    # 训练、验证、checkpoint、CSV/JSON/TensorBoard
analyze.py                  # 曲线、汇总 CSV/JSON、高清 PNG
verify_checkpoint.py        # best/final 严格加载、张量一致性与 SHA-256 核验
tests/                      # 契约与无泄漏测试
results/                    # 提交的汇总 CSV/JSON
figures/                    # 提交的高清曲线/图
REPORT.md                   # 实验报告
STATUS.md                   # 当前进度、checkpoint 与指标摘要（续接用）
```

## 运行

```bash
# Smoke（64 条 Train-Seen / 32 条 Val，20 步）
python experiments/visual_encoder_citynav/train.py \
  --smoke --run-name smoke --device cuda --num-workers 0

# 正式全量 Train-Seen（22,002 条轨迹，单 epoch）
python experiments/visual_encoder_citynav/train.py \
  --run-name full_YYYYMMDD --device cuda --num-workers 2 --val-samples 256 --batch-size 8

# 生成曲线/汇总
python experiments/visual_encoder_citynav/analyze.py --run-name full_YYYYMMDD

# 核验 best/final checkpoint 可严格加载且张量一致
python experiments/visual_encoder_citynav/verify_checkpoint.py \
  --run-name full_YYYYMMDD
```

## 本地产物

- 点云 downsample：`data_cache/visual_encoder_citynav/clouds/`
- manifest / BERT 特征：`data_cache/visual_encoder_citynav/manifests/`
- 固定候选库：`data_cache/candidates/`
- 训练产物：`runs/visual_encoder_citynav/<run>/`
- best / final encoder：`runs/visual_encoder_citynav/<run>/best_encoder.pt`、
  `encoder_final.pt`
- TensorBoard：`runs/visual_encoder_citynav/<run>/tensorboard`

以上均在 `.gitignore` 范围内，不提交。GitHub 只提交实验代码、配置、协议、
汇总 CSV/JSON、报告和结果图。

## 初始化

默认从 `data_cache/weights/shared_encoder.pt` 严格加载；缺失/键或形状不兼容会
直接失败，不会静默随机初始化。预期加载比例 1.0000（3,997,056 参数）。

## 测试

```bash
pytest -q experiments/visual_encoder_citynav/tests
```

## 限制

- FPV 来自只读 PLY 的 4M 点 downsample，存在空洞；
- 不保证与 AirSim 完全一致；
- 只在 Train-Seen 上单 epoch，报告中的动作/匹配指标是辅助监督诊断，不是
  Small/Transformer-M MEC 的导航成功率。
