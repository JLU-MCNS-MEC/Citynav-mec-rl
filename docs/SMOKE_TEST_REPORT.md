# CityNav-MEC FPV 六动作 Smoke Test 报告

- 日期：2026-10-27
- 环境：Python 3.10.21，torch 2.9.1+cu128
- Smoke 设备：CPU（正式过拟合验证使用 RTX 5090）
- 数据：只读 CityNav / CityRefer / SensatUrban PLY；未读取 `test_unseen`

## 1. Smoke Test

命令：

```bash
python scripts/smoke_test.py \
  --split val_unseen \
  --episodes 1 \
  --observation fpv \
  --render-size 336 \
  --crop-radius 120 \
  --splat-radius 0 \
  --device cpu \
  --text-backend hashing
```

说明：Smoke Test 为快速离线链路检查，显式使用 hashing 文本后端；**正式训练配置使用冻结 BERT**，见 `docs/OVERFIT_REPORT.md`。

| 项目 | 结果 |
|---|---|
| 观测 | 真实 SensatUrban PLY 第一视角 RGB + 前向 Depth |
| 视觉编码器 | 576 × 384 tokens |
| 目标参考 | 3 × 96 × 96 × 3 |
| Heatmap | 3 × 8 × 8 |
| Small | 548,086 参数，前向 [1, 6]，train loss 2.4460，grad norm 10.7587 |
| Transformer-M | 2,822,006 参数，前向 [1, 6]，train loss 1.9837，grad norm 28.9872 |
| 原始数据快照 | 未变化，摘要 `2c784870be858d7d801ee7c12e68c81b44b11cb7` |

Smoke 只做一次前向和一次优化步，batch=1，不能用于判断准确率。

## 2. 测试

```bash
pytest -q
# 50 passed
```

覆盖：

- 六动作官方 ID/顺序、STOP 5 m 教师阈值与 20 m 评估成功半径分离；
- 真实高度保留、六元组 yaw/pitch、退化方向回退；
- 五类移动头 + 二元 Stop 头的六类组合；
- 损失、单步参数更新、per-class 指标；
- 第一视角针孔投影/深度 z-buffer/PLY memmap；
- 冻结本地 BERT（参数 requires_grad=False、无梯度）；
- 真实 val_unseen FPV 观测与 Small 前向；
- Git 根目录边界。

## 3. 本轮修复/变更

- 动作空间改为官方六类：`STOP / MOVE_FORWARD / TURN_RIGHT / TURN_LEFT / GO_UP / GO_DOWN`；
- 移动头 3→5 类，`combine_move_and_arrival` 输出 [Stop, Forward, Right, Left, Up, Down]；
- 教师改为官方 `interpolated_trajectory + marker`，lookahead=1，教师 STOP 5 m，评估成功 20 m；
- 删除 50 m 固定高度，保留真实轨迹 z；
- 新增 SensatUrban 原始 PLY 第一视角渲染器、FPV 观测适配器、样例/过拟合脚本；
- 文本默认改为冻结 BERT；hashing 仅保留离线测试；
- 视觉编码器可加载 MEC-V8.2 `shared_encoder.pt` 并参与训练。

## 4. 限制

- Smoke 使用 hashing，不是正式训练成绩；正式过拟合验证见 `docs/OVERFIT_REPORT.md`；
- PLY 点云渲染有空洞、无表面重建、无光照，不等于 AirSim；
- 未做完整训练、val 指标评测、RL 或 GSM；
- 网络状态为占位 5 维向量。
