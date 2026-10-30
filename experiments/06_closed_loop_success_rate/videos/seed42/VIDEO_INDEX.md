# Stage 06 闭环飞行视频索引

共 **18** 条真实闭环 episode：12 条成功、6 条失败诊断对照。

全部视频合计 33.2 MB。

> 视频是**离散动作回放**：每步 3 帧、10 fps，按住真实状态与动作顺序播放，不是真实物理时间录像。视觉帧来自 Stage 06 的 `heading_crop()`。

## 成功案例

### val_seen / short

- `val_seen:00060` (birmingham_block_1, 19 steps, 6.9s): [MP4](videos/seed42/success/val_seen/short/val_seen_00060.mp4) · [封面](videos/seed42/covers/val_seen_00060.png)
- `val_seen:00338` (birmingham_block_3, 18 steps, 6.6s): [MP4](videos/seed42/success/val_seen/short/val_seen_00338.mp4) · [封面](videos/seed42/covers/val_seen_00338.png)

### val_seen / medium

- `val_seen:01259` (cambridge_block_9, 39 steps, 12.9s): [MP4](videos/seed42/success/val_seen/medium/val_seen_01259.mp4) · [封面](videos/seed42/covers/val_seen_01259.png)
- `val_seen:00937` (cambridge_block_7, 31 steps, 10.5s): [MP4](videos/seed42/success/val_seen/medium/val_seen_00937.mp4) · [封面](videos/seed42/covers/val_seen_00937.png)

### val_seen / long

- `val_seen:00504` (birmingham_block_4, 56 steps, 18.0s): [MP4](videos/seed42/success/val_seen/long/val_seen_00504.mp4) · [封面](videos/seed42/covers/val_seen_00504.png)
- `val_seen:00709` (birmingham_block_9, 60 steps, 19.2s): [MP4](videos/seed42/success/val_seen/long/val_seen_00709.mp4) · [封面](videos/seed42/covers/val_seen_00709.png)

### val_unseen / short

- `val_unseen:00022` (birmingham_block_5, 27 steps, 9.3s): [MP4](videos/seed42/success/val_unseen/short/val_unseen_00022.mp4) · [封面](videos/seed42/covers/val_unseen_00022.png)
- `val_unseen:02380` (cambridge_block_3, 14 steps, 5.4s): [MP4](videos/seed42/success/val_unseen/short/val_unseen_02380.mp4) · [封面](videos/seed42/covers/val_unseen_02380.png)

### val_unseen / medium

- `val_unseen:01328` (cambridge_block_10, 43 steps, 14.1s): [MP4](videos/seed42/success/val_unseen/medium/val_unseen_01328.mp4) · [封面](videos/seed42/covers/val_unseen_01328.png)
- `val_unseen:01968` (cambridge_block_2, 37 steps, 12.3s): [MP4](videos/seed42/success/val_unseen/medium/val_unseen_01968.mp4) · [封面](videos/seed42/covers/val_unseen_01968.png)

### val_unseen / long

- `val_unseen:00976` (cambridge_block_10, 50 steps, 16.2s): [MP4](videos/seed42/success/val_unseen/long/val_unseen_00976.mp4) · [封面](videos/seed42/covers/val_unseen_00976.png)
- `val_unseen:02662` (cambridge_block_10, 56 steps, 18.0s): [MP4](videos/seed42/success/val_unseen/long/val_unseen_02662.mp4) · [封面](videos/seed42/covers/val_unseen_02662.png)

## 失败诊断案例

- `val_seen:00005` — premature_stop (val_seen/medium, birmingham_block_1, 35 steps): [MP4](videos/seed42/failure/val_seen/medium/val_seen_00005.mp4) · [封面](videos/seed42/covers/val_seen_00005.png)
- `val_seen:00003` — reached_but_not_stopped (val_seen/medium, birmingham_block_1, 38 steps): [MP4](videos/seed42/failure/val_seen/medium/val_seen_00003.mp4) · [封面](videos/seed42/covers/val_seen_00003.png)
- `val_seen:00001` — long_straight_oob (val_seen/long, birmingham_block_1, 83 steps): [MP4](videos/seed42/failure/val_seen/long/val_seen_00001.mp4) · [封面](videos/seed42/covers/val_seen_00001.png)
- `val_seen:00056` — turn_oscillation (val_seen/medium, birmingham_block_1, 53 steps): [MP4](videos/seed42/failure/val_seen/medium/val_seen_00056.mp4) · [封面](videos/seed42/covers/val_seen_00056.png)
- `val_seen:00383` — low_valid_target (val_seen/medium, birmingham_block_3, 81 steps): [MP4](videos/seed42/failure/val_seen/medium/val_seen_00383.mp4) · [封面](videos/seed42/covers/val_seen_00383.png)
- `val_unseen:02749` — collision (val_unseen/medium, cambridge_block_2, 56 steps): [MP4](videos/seed42/failure/val_unseen/medium/val_unseen_02749.mp4) · [封面](videos/seed42/covers/val_unseen_02749.png)

## 汇总

- 12 条成功轨迹汇总视频：[success_compilation.mp4](success_compilation.mp4) (12.4 MB)
- 封面拼图：[video_montage.png](video_montage.png)
- 清单：[video_manifest.csv](video_manifest.csv)（含每个 MP4 的 SHA-256）
