# CityNav-MEC 数据适配与 Smoke Test 报告

- 日期：2026-10-27
- 环境：Python 3.10.21，torch 2.9.1+cu128，CPU（Smoke Test 未占用 GPU）
- 数据：只读 CityNav / CityRefer / RGBD / SensatUrban PLY 共享目录
- 范围：数据读取与字段检查、官方 top-down 观测、视觉/语言编码、
  Small 与 Transformer-M 前向、单步监督训练。不包含 GSM、RL 环境或完整训练。

## 1. 数据审计

命令：

```bash
python scripts/audit_dataset.py \
  --splits train_seen val_seen val_unseen \
  --teacher-sample 100 \
  --output-dir reports
```

结果：

| split | 记录数 | 预期 | 地图数 | 描述严格匹配 | 描述归一化匹配 | 目标越界 | 重复 episode id |
|---|---:|---:|---:|---:|---:|---:|---:|
| train_seen | 22,002 | 22,002 | 24 | 21,463 | 22,002 | 0 | 0 |
| val_seen | 2,498 | 2,498 | 24 | 2,430 | 2,498 | 0 | 0 |
| val_unseen | 2,826 | 2,826 | 4 | 2,757 | 2,826 | 0 | 0 |

其他检查：

- CityRefer：34 个 block、26,274 个对象；三个开发划分引用 28 个 block，
  `test_unseen` 的 6 个 block 未读取；
- 所有被引用地图均具备 PNG、TIF、PLY；无缺失资产；
- PNG 与 TIF 尺寸逐图一致，地理变换均为 0.1 m；由于 block 裁剪，
  尺寸不全是 4000×4000；
- PLY 只读取文件头和 16 个顶点；28 个开发地图头部合计 1,770,136,726 个顶点，
  均有 `x/y/z`，样本坐标有限；
- 教师回放抽样 100 条/划分，成功率均为 1.000；终点距离中位数约
  17.6–18.1 m（成功半径 20 m），交叉轨迹中位数约 3.9–4.1 m；
- 运行前后原始数据快照一致（文件数 135，摘要
  `2c784870be858d7d801ee7c12e68c81b44b11cb7`）。

### 描述匹配的差异来源

严格字符串比较约 97.4% 匹配。逐条核对后，差异只有两类：

1. 连续空格/空白数量不同；
2. CityRefer 原文用 ASCII 双引号包住字母或短语（如 `"L"`、`"The Gym"`），
   CityNav split 文本去掉了引号，且部分引号紧邻逗号/句号。

`ann_ids` 始终是对象 `descriptions` 列表的 0-based 索引；归一化空白并删除
ASCII 双引号后，三个开发划分均为 100% 匹配。语言输入仍使用 split 原文，
不做文本改写。

## 2. Smoke Test

命令：

```bash
python scripts/smoke_test.py \
  --split val_unseen --episodes 2 --device cpu --text-backend hashing
```

使用 episode：

| episode | map | teacher steps | success | description match |
|---|---|---:|---:|---:|
| `0b2c9070292886d602bf` | birmingham_block_5 | 39 | True | True |
| `09fb05b13d3a191c2add` | birmingham_block_5 | 96 | True | True |

模型结果（batch=2，单步 AdamW）：

| model | 参数量 | 前向形状 | eval loss | train loss | grad norm | 参数变化 L2 |
|---|---:|---|---:|---:|---:|---:|
| Small | 547,572 | [2, 4] | 1.768783 | 1.803192 | 5.504662 | 0.067056 |
| Transformer-M | 2,821,492 | [2, 4] | 2.103695 | 2.150671 | 22.224602 | 0.158745 |

检查项全部通过：

- 视觉编码器输出 `[B, 576, 384]`；
- 目标参考输出 `[B, 3, 384]`，热图 `[B, 3, 8, 8]`；
- 4 动作联合分布归一化；
- loss、梯度、参数更新均为有限值；
- 原始数据摘要运行前后一致。

## 3. 测试

```bash
pytest -q
# 41 passed
```

唯一输出是 PyTorch 的 `enable_nested_tensor` 性能提示，不影响数值或测试结果。

## 4. 本轮修复

- `src/training/supervised.py`：训练步不再把 `actions` 透传给
  `forward_components`；此前的实现会直接 `TypeError`，无法完成一步训练。
  同时让标签自动对齐 logits 所在设备。
- `src/data/trajectory_adapter.py`：修复 `_motion_yaw` 使用前一个点时的
  方向符号错误。此前退化方向的 yaw 回退会反转 180°。
- `src/data/observation_adapter.py`：小 PLY 的顶点采样按实际顶点数截断，
  避免审计时对顶点数少于 16 的文件误报。
- `scripts/audit_dataset.py`：默认不读取 `test_unseen`；新增
  `--include-test-unseen` 显式开关；直接构造 CityReferIndex，避免为了拿对象表
  而加载一个 split。
- `src/data/citynav_dataset.py`：增加 split 字段一致性校验、描述严格/归一化
  匹配统计。
- 新增测试、Smoke 脚本、README 和本报告。

## 5. 限制与下一步

- 观测是官方 top-down 正射裁剪，不是第一人称 RGB；`render_fpv_heightfield`
  只是 2.5-D 透视代理；PLY 第一人称渲染尚未实现；
- `depth` 通道是官方约定的离地高度代理，不是视线深度；
- 网络状态是确定性占位 5-D 向量，不是实测 20-MEC 状态；
- 文本编码器冻结；当前只验证了离线 hashing 后端；
- 尚未移植 V8.2 的完整训练配方（movement class balance、arrival hard states、
  checkpoint 指标）；
- 下一步是在 train_seen 上跑完整监督基线并在 val_seen / val_unseen 评测
  SR / SPL / NE，再决定 RL 与真实网络状态接入。
