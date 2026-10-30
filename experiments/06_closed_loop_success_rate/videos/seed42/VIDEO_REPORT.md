# Stage 06 闭环飞行视频报告

## 1. 目的

把 Stage 06 保存的真实闭环轨迹渲染成可观看视频，用于人工检查模型行为：是否朝目标飞、何时转向、何时 STOP、失败发生在什么位置。

**没有重新运行模型、没有重新生成轨迹、没有修改任何动作或概率。**每帧都来自 `results/seed42/episodes/<id>.json` 的原始逐步记录。

## 2. 生成方式

- 左：North-up 真实 RGB 地图（数据 `DATA/rgbd/`），绘制 20 m 成功圈、起点、目标、实际轨迹、当前位置与航向。
- 右：当前位姿的 `MapRaster.heading_crop()` 局部俯视 224×224（放大 2 倍），与模型输入完全一致。
- 底部：episode id、划分、长度组、step、动作、目标距离、四类概率、最终结果。
- 编码：H.264 (`libx264`)，1280×720，10 fps，crf 26；每步保持 3 帧。

## 3. 案例选择

成功案例：每个（划分 × 长度组）2 条，优先选择地图不同、转弯次数较多的任务；失败案例：按失败类型各 1 条，包括过早 STOP、进入 20 m 未停、长任务零转向越界、转向振荡、低有效目标图像和碰撞。

| episode | 类型 | 划分 | 长度 | 地图 | 步数 | 时长 | 文件 |
|---|---|---|---|---|---|---|---|
| val_seen:00060 | success | val_seen | short | birmingham_block_1 | 19 | 6.9s | `val_seen_00060.mp4` |
| val_seen:00338 | success | val_seen | short | birmingham_block_3 | 18 | 6.6s | `val_seen_00338.mp4` |
| val_seen:01259 | success | val_seen | medium | cambridge_block_9 | 39 | 12.9s | `val_seen_01259.mp4` |
| val_seen:00937 | success | val_seen | medium | cambridge_block_7 | 31 | 10.5s | `val_seen_00937.mp4` |
| val_seen:00504 | success | val_seen | long | birmingham_block_4 | 56 | 18.0s | `val_seen_00504.mp4` |
| val_seen:00709 | success | val_seen | long | birmingham_block_9 | 60 | 19.2s | `val_seen_00709.mp4` |
| val_unseen:00022 | success | val_unseen | short | birmingham_block_5 | 27 | 9.3s | `val_unseen_00022.mp4` |
| val_unseen:02380 | success | val_unseen | short | cambridge_block_3 | 14 | 5.4s | `val_unseen_02380.mp4` |
| val_unseen:01328 | success | val_unseen | medium | cambridge_block_10 | 43 | 14.1s | `val_unseen_01328.mp4` |
| val_unseen:01968 | success | val_unseen | medium | cambridge_block_2 | 37 | 12.3s | `val_unseen_01968.mp4` |
| val_unseen:00976 | success | val_unseen | long | cambridge_block_10 | 50 | 16.2s | `val_unseen_00976.mp4` |
| val_unseen:02662 | success | val_unseen | long | cambridge_block_10 | 56 | 18.0s | `val_unseen_02662.mp4` |
| val_seen:00005 | failure/premature_stop | val_seen | medium | birmingham_block_1 | 35 | 11.7s | `val_seen_00005.mp4` |
| val_seen:00003 | failure/reached_but_not_stopped | val_seen | medium | birmingham_block_1 | 38 | 12.6s | `val_seen_00003.mp4` |
| val_seen:00001 | failure/long_straight_oob | val_seen | long | birmingham_block_1 | 83 | 26.1s | `val_seen_00001.mp4` |
| val_seen:00056 | failure/turn_oscillation | val_seen | medium | birmingham_block_1 | 53 | 17.1s | `val_seen_00056.mp4` |
| val_seen:00383 | failure/low_valid_target | val_seen | medium | birmingham_block_3 | 81 | 25.5s | `val_seen_00383.mp4` |
| val_unseen:02749 | failure/collision | val_unseen | medium | cambridge_block_2 | 56 | 18.0s | `val_unseen_02749.mp4` |

## 4. 体量与许可

- 全部 MP4 合计约 33.2 MB，最大单文件 2.1 MB，低于 GitHub 100 MB 限制。
- 视频由本项目自己的地图数据与模型轨迹渲染，不包含第三方版权素材。

## 5. 限制

- 离散动作回放，不是实时飞行录像；帧数 = 步数 × 3 + 尾部停留。
- 局部视图是 `heading_crop` 的静态透视裁剪，不含渲染/光照。
- 视频只用于人工定性检查；定量结论以 Stage 06 的指标表为准。
