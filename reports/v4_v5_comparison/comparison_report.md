# V4 / V5 修正版闭环对比报告

## 1. 总表（统一协议：SR/OSR 20 m，NE=最终距离，SPL=成功率加权路径比，max steps=200）

| 指标 | V4 | V5 | 差异 |
|---|---:|---:|---:|
| Val-Seen SR | 0.4712 | 0.6025 | +0.1313 |
| Val-Unseen SR | 0.4144 | 0.5311 | +0.1168 |
| Test-Unseen SR | 0.4063 | 0.5261 | +0.1198 |
| Test OSR | 0.4267 | 0.5404 | +0.1137 |
| Test NE (m) | 104.0 | 67.0 | -37.0 |
| Test SPL | 0.3604 | 0.4317 | +0.0713 |
| Short SR | 0.5272 | 0.6272 | +0.1000 |
| Medium SR | 0.3977 | 0.5273 | +0.1296 |
| Long SR | 0.4042 | 0.5437 | +0.1395 |
| 平均导航步数 | 75.9 | 89.0 | +13.2 |
| P95 导航步数 | 200 | 200 | +0 |

## 2. 逐 episode 对齐

- 共同任务：10635
- 双成功：3636 | 仅 V4 成功：870 | 仅 V5 成功：2164 | 双失败：3965

## 3. 分析

1. 几何增益：Test-Unseen SR 差异 +0.1198（多地标几何带来增益）
2. 空候选任务（block_32 等）：V4 220 条 SR=0.0 | V5 220 条 SR=0.08181818181818182
3. 失败构成 V4：{'out_of_bounds': 4521, 'premature_stop': 704, 'reached_but_not_stopped': 192, 'never_stop': 708, 'collision': 4}
4. 失败构成 V5：{'out_of_bounds': 1515, 'never_stop': 1799, 'premature_stop': 1357, 'reached_but_not_stopped': 156, 'collision': 8}
5. 长距离任务（Long）：V4 0.4042 → V5 0.5437（+0.1395）
6. 路径效率（SPL）：V4 0.3747 → V5 0.4474
7. 早停：V4 premature_stop 704 条 | V5 1357 条

## 4. 历史基线（旧协议，非直接可比）

- C1 (09C-1) val SR：0.647
- M-v3 (10E) val SR：0.554

## 5. 复现信息

- 评测代码：experiments/v4_closed_loop_revalidation/scripts/run_closed_loop.py
- V4 checkpoint SHA256：c90fd9d60d27d65a2947e01ff8f0e76f1925863367d30f8d9f14cab222e08e05
- V5 checkpoint SHA256：11ad6692bfcbfd793de1ecab296f7dff8cc5f3c56f7d3579fea07b5948e45416
- 任务清单：Stage 06 manifests（val 5,324 + test 5,311），全部任务执行，SR 分母=全部有效任务
- 旧版 11D 结果（history 顺序/Pose 编码带病）标记为 historical/incomplete

