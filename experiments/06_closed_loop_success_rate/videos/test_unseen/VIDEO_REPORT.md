> **范围说明**：本目录是 **Test-Unseen 追加评测**（5,311 条）的 11 条案例视频，不是 Stage 06 验证集的 18 条。总述见 [`../../TEST_UNSEEN_ADDENDUM.md`](../../TEST_UNSEEN_ADDENDUM.md)。

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
| test_unseen:00318 | success | test_unseen | None | birmingham_block_0 | 20 | 7.2s | `test_unseen_00318.mp4` |
| test_unseen:00567 | success | test_unseen | None | birmingham_block_6 | 17 | 6.3s | `test_unseen_00567.mp4` |
| test_unseen:01128 | success | test_unseen | None | cambridge_block_14 | 21 | 7.5s | `test_unseen_01128.mp4` |
| test_unseen:01828 | success | test_unseen | None | cambridge_block_26 | 11 | 4.5s | `test_unseen_01828.mp4` |
| test_unseen:03230 | success | test_unseen | None | cambridge_block_32 | 1 | 1.5s | `test_unseen_03230.mp4` |
| test_unseen:03663 | success | test_unseen | None | cambridge_block_8 | 19 | 6.9s | `test_unseen_03663.mp4` |
| test_unseen:00002 | failure/premature_stop | test_unseen | None | birmingham_block_0 | 23 | 8.1s | `test_unseen_00002.mp4` |
| test_unseen:03228 | failure/zero_belief_stop | test_unseen | None | cambridge_block_32 | 1 | 1.5s | `test_unseen_03228.mp4` |
| test_unseen:00000 | failure/out_of_bounds | test_unseen | None | birmingham_block_0 | 56 | 18.0s | `test_unseen_00000.mp4` |
| test_unseen:00008 | failure/reached_but_not_stopped | test_unseen | None | birmingham_block_0 | 15 | 5.7s | `test_unseen_00008.mp4` |
| test_unseen:02536 | failure/collision | test_unseen | None | cambridge_block_26 | 28 | 9.6s | `test_unseen_02536.mp4` |

## 4. 体量与许可

- 全部 MP4 合计约 9.0 MB，最大单文件 1.2 MB，低于 GitHub 100 MB 限制。
- 视频由本项目自己的地图数据与模型轨迹渲染，不包含第三方版权素材。

## 5. 限制

- 离散动作回放，不是实时飞行录像；帧数 = 步数 × 3 + 尾部停留。
- 局部视图是 `heading_crop` 的静态透视裁剪，不含渲染/光照。
- 视频只用于人工定性检查；定量结论以 Stage 06 的指标表为准。
