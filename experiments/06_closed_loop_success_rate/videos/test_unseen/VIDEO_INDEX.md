# Stage 06 闭环飞行视频索引

共 **11** 条真实闭环 episode：6 条成功、5 条失败诊断对照。

全部视频合计 9.0 MB。

> 视频是**离散动作回放**：每步 3 帧、10 fps，按住真实状态与动作顺序播放，不是真实物理时间录像。视觉帧来自 Stage 06 的 `heading_crop()`。

## 成功案例

## 失败诊断案例

- `test_unseen:00002` — premature_stop (test_unseen/None, birmingham_block_0, 23 steps): [MP4](videos/test_unseen/failure/test_unseen/birmingham_block_0/test_unseen_00002.mp4) · [封面](videos/test_unseen/covers/test_unseen_00002.png)
- `test_unseen:03228` — zero_belief_stop (test_unseen/None, cambridge_block_32, 1 steps): [MP4](videos/test_unseen/failure/test_unseen/cambridge_block_32/test_unseen_03228.mp4) · [封面](videos/test_unseen/covers/test_unseen_03228.png)
- `test_unseen:00000` — out_of_bounds (test_unseen/None, birmingham_block_0, 56 steps): [MP4](videos/test_unseen/failure/test_unseen/birmingham_block_0/test_unseen_00000.mp4) · [封面](videos/test_unseen/covers/test_unseen_00000.png)
- `test_unseen:00008` — reached_but_not_stopped (test_unseen/None, birmingham_block_0, 15 steps): [MP4](videos/test_unseen/failure/test_unseen/birmingham_block_0/test_unseen_00008.mp4) · [封面](videos/test_unseen/covers/test_unseen_00008.png)
- `test_unseen:02536` — collision (test_unseen/None, cambridge_block_26, 28 steps): [MP4](videos/test_unseen/failure/test_unseen/cambridge_block_26/test_unseen_02536.mp4) · [封面](videos/test_unseen/covers/test_unseen_02536.png)

## 汇总

- 6 条成功轨迹汇总视频：[success_compilation.mp4](success_compilation.mp4) (3.2 MB)
- 封面拼图：[video_montage.png](video_montage.png)
- 清单：[video_manifest.csv](video_manifest.csv)（含每个 MP4 的 SHA-256）
