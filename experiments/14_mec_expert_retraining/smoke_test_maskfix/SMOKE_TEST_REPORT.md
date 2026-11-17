# Stage 14-R4 真实数据冒烟测试报告

日期：2026-11-17

每个模型最多 100 个真实训练 micro-batch；Large 为 batch 64 × 累积 2。
**100 steps 不代表收敛**；本报告只验证数据/前向/反向/优化器/验证/
Checkpoint 链路。

| 模型 | 参数 | Batch×累积 | 更新步 | Loss first→last10 | Val Macro-F1 | Grad norm mean/max | Step ms | Peak GB | NaN/Inf | Ckpt MB | Resume |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Transformer-S | 211,181,060 | 128×1 | 100 | 1.6788→0.8356 | 0.1590 | 5.888/40.554 | 151.1 | 10.08 | 0 | 2534 | OK |
| Large | 1,034,877,444 | 64×2 | 50 | 1.0519→0.6874 | 0.1592 | 14.695/60.055 | 271.6 | 28.29 | 0 | 12419 | OK |

## 判定

- Transformer-S: PASS (NaN/Inf=0, F1=0.1590, resume=True, peak=10.08GB)
- Large: PASS (NaN/Inf=0, F1=0.1592, resume=True, peak=28.29GB)

总体：全部通过。

产物：`results/<model>/{history.jsonl,summary.json}`、
`checkpoints/<model>.pt`。未训练完整 epoch，未跑闭环/DQN/PPO。
