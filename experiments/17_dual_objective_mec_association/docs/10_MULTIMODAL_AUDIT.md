# Stage 17-A-R2 · 多模态数据部署与上行负载审计

审计方式：逐项对照 `env16.EpisodeEnv16.step` 构造的 batch 与 `models.py` 的 forward 输入，
以及 `run_closed_loop.py` 的 `belief_features / landmark_inputs / history_inputs`。
**未运行环境**，形状与 dtype 来自源码常量与函数返回类型。

## 1. 模态清单

| # | 模态 | shape | dtype | 元素数 | 生成位置 | MEC 需要 | 每步上传 | 可缓存 |
|---|---|---|---:|---:|---|---|---|---|
| 1 | Current Visual | (49, 2048) | float32 | 100,352 | UAV FrozenEncoder (layer4, fp16 计算 → fp32 输出) | ✅ | ✅ | ❌（每步变化） |
| 2 | Target Embedding | (512,) | float32 | 512 | UAV layer2 目标描述子，每 episode 一次 | ✅ | ✅ | 理论可（v1 仍计费） |
| 3 | History Visual | (16, 2048) | float32 | 32,768 | UAV，前 16 步 layer4 池化 | ✅ | ✅ | ❌（每步重建） |
| 4 | History Pose | (16, 3) | float32 | 48 | UAV | ✅ | ✅ | ❌ |
| 5 | History Action | (16, 4) | float32 | 64 | UAV，one-hot | ✅ | ✅ | ❌ |
| 6 | History Mask | (16,) | float32 | 16 | UAV | ✅ | ✅ | ❌ |
| 7 | Belief | (10,) | float32 | 10 | UAV，`belief_features()` | ✅ | ✅ | ❌ |
| 8 | Match | (5,) | float32 | 5 | UAV，`subpixel_match()` | ✅ | ✅ | ❌ |
| 9 | Landmark | (8, 3) | float32 | 24 | UAV，`landmark_inputs()` | ✅ | ✅ | ❌ |
| 10 | Relation | (1, 8, 3) | float32 | 24 | **MEC**，`TransformerM12Fix.landmark_geometry(landmarks, states)` | ✅ | **❌ 不上传** | — |
| 11 | UAV State | (3,) | float32 | 3 | UAV | ✅ | ✅ | ❌ |

> **不变量**：Relation 由 MEC 从 Landmark + UAV State 现场推导，若再上传会造成重复计费，
> 因此 `payload_bits(relation)=0`。

## 2. 通信精度 vs 计算精度

FrozenEncoder 的 ResNet-50 以 **fp16** 运行，但输出特征为 **float32**。
通信序列化精度是独立选择：v1 采用 **FP32（32 bit/元素）**，不引入未经验证的 FP16 压缩。
配置项 `feature_bits` 可改，但改动后必须重新审核负载与时延。

## 3. 上行负载（逐步）

| 模态 | 元素数 | bit/元素 | 负载 (bit) |
|---|---:|---:|---:|
| Current Visual | 100,352 | 32 | 3,211,264 |
| Target Embedding | 512 | 32 | 16,384 |
| History Visual | 32,768 | 32 | 1,048,576 |
| History Pose | 48 | 32 | 1,536 |
| History Action | 64 | 32 | 2,048 |
| History Mask | 16 | 32 | 512 |
| Belief | 10 | 32 | 320 |
| Match | 5 | 32 | 160 |
| Landmark | 24 | 32 | 768 |
| UAV State | 3 | 32 | 96 |
| Relation | 24 | — | **0（不上传）** |
| **合计** | | | **4,281,664 bit = 535,208 B** |

对比 Stage 16：1,605,632 bit（仅 Current Visual），R2 为 **2.67 倍**。

## 4. 缓存策略

`caching_policy = "none"`：**不假设任何模态已预置在 MEC 上**。
Target Embedding 每 episode 恒定，理论上可在 episode 开始时下发一次，
但该协议（建立成本、失效规则、MEC 间一致性）尚未实现，
因此 v1 **每步都计费**。若后续引入缓存，必须：
1. 定义缓存建立动作与它的传输成本；
2. 定义失效条件（任务切换）；
3. 在日志中记录 `cache_hit`。

当前实现中 `calculate_uplink_payload_bits(..., "static_cache")` 会显式抛
`NotImplementedError`，不会静默返回更小的负载。

## 5. 数据放置

- **UAV 端**：RGB 观测、layer4 编码、layer2 匹配编码、belief/match、16 步历史、目标描述子。
- **MEC 端**：仅持有冻结导航专家参数；接收上传特征后按模型协议做一次前向；
  **不重新执行视觉编码**。
- **MEC 之间**：不新增历史状态迁移协议；每次请求携带完整动态输入，避免切换时缺历史。

## 6. 与 Stage 16 的差异

| 项目 | Stage 16 | Stage 17-A-R2 |
|---|---|---|
| 上行负载 | 1,605,632 bit（仅 current） | 4,281,664 bit（11 模态） |
| 通信几何 | 80/10 m | 50/10 m |
| UAV 本地计算 | 只有 layer4（5.45 ms） | layer4 + layer2 + matching（7.66 ms，matching 未核定） |
| 下行 | 无 | 无（显式忽略） |
