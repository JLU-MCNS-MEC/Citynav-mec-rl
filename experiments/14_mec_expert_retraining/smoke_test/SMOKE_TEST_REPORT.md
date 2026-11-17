# Stage 14-R4 真实数据冒烟测试报告

日期：2026-11-17

每个模型最多 100 个真实训练 micro-batch；Large 为 batch 64 × 累积 2。
**100 steps 不代表收敛**；本报告只验证数据/前向/反向/优化器/验证/
Checkpoint 链路。

| 模型 | 参数 | Batch×累积 | 更新步 | Loss first→last10 | Val Macro-F1 | Grad norm mean/max | Step ms | Peak GB | NaN/Inf | Ckpt MB | Resume |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Small | 61,121,028 | 128×1 | 100 | 2.6334→0.8454 | 0.3182 | 60.021/368.545 | 26.8 | 1.64 | 0 | 733 | OK |
| Medium | 166,011,908 | 128×1 | 100 | 2.6287→0.8278 | 0.2544 | 57.953/274.272 | 45.1 | 4.04 | 0 | 1992 | OK |
| Transformer-S | 211,181,060 | 128×1 | 100 | 1.4041→0.8366 | 0.1592 | 5.226/41.621 | 147.3 | 10.08 | 0 | 2534 | OK |
| Transformer-M | 498,567,684 | 128×1 | 100 | 1.2510→0.8347 | 0.1592 | 7.141/40.164 | 291.4 | 19.19 | 0 | 5983 | OK |
| Large | 1,034,877,444 | 64×2 | 50 | 1.2380→0.6980 | 0.1592 | 10.486/47.301 | 268.1 | 28.28 | 0 | 12419 | OK |

## 判定

- Small: PASS (NaN/Inf=0, F1=0.3182, resume=True, peak=1.64GB)
- Medium: PASS (NaN/Inf=0, F1=0.2544, resume=True, peak=4.04GB)
- Transformer-S: PASS (NaN/Inf=0, F1=0.1592, resume=True, peak=10.08GB)
- Transformer-M: PASS (NaN/Inf=0, F1=0.1592, resume=True, peak=19.19GB)
- Large: PASS (NaN/Inf=0, F1=0.1592, resume=True, peak=28.28GB)

总体：全部通过。

产物：`results/<model>/{history.jsonl,summary.json}`、
`checkpoints/<model>.pt`。未训练完整 epoch，未跑闭环/DQN/PPO。
