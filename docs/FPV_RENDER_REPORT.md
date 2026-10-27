# 第一视角 PLY RGB/Depth 渲染报告

- 日期：2026-10-27
- 数据：只读 SensatUrban 原始 PLY + CityNav 原始轨迹 / CityRefer
- 渲染器：新仓库独立实现的 CPU 点云针孔渲染器，不是 AirSim
- `test_unseen`：未读取

## 1. 相机与坐标定义

| 项目 | 定义 |
|---|---|
| 坐标 | CityNav / SensatUrban 局部坐标，`x/y` 水平米，`z` 向上 |
| 世界变换 | 恒等 XY/Z；抽检 block 的 PLY/TIF XY 包围盒差约 ≤ 0.1 m |
| yaw | `atan2(dy, dx)`；正 yaw 向 `+y`（左）转 |
| pitch | `atan2(dz, hypot(dx, dy))`；负 pitch 向下 |
| 相机基 | yaw=0,pitch=0 时 forward=+x，right=-y，up=+z |
| 图像 | 512×512（样例）；训练缓存 336×336 |
| HFOV/VFOV | 90° / 90°（方形图） |
| 内参 | `fx=fy=W/(2 tan 45°)=W/2`，`cx=cy=(W-1)/2` |
| 主图像素 | 512: `fx=fy=256`，`cx=cy=255.5` |
| 近/远裁剪 | 0.5 m / 300 m |
| Depth | 相机前向深度 `dot(point-camera, forward)`，单位 m；不是欧氏距离，也不是俯视高度图 |
| 相机俯仰 | 正式路径使用 CityNav 方向向量的 pitch（`fpv_pitch_mode=trajectory`）；`fixed` 仅用于诊断 |
| 点云颜色 | PLY 的 `red/green/blue`，无光照/纹理重建 |

## 2. 渲染流程

1. `open_ply_cloud()` 用 `np.memmap` 读取原始 binary PLY，不复制原始数据；
2. `crop_ply_cloud()` 只取指定半径内的 XYZRGB 点；
3. 按相机基投影到针孔平面，使用逐像素最小深度 z-buffer；
4. 输出 RGB、前向 depth、有效 mask、每像素点数；
5. 目标/教师 marker 用同一相机模型投影叠加。

没有做表面重建、纹理插值、光照或遮挡补全，因此**空洞和点状稀疏是预期现象**。这不是 AirSim 渲染，不能声称与官方 AirSim 帧一致。

## 3. 样例

`figures/fpv_examples/` 包含：

- 4 个真实第一视角 RGB（val_seen 3 个不同地图 + val_unseen birmingham_block_5）；
- 4 个对应 depth 可视化（TURBO，黑色为无有效点）；
- 4 个 target/marker 投影叠加图；
- `contact_sheet.jpg`；
- `topdown_vs_ply.jpg`：同一位姿 pitch=-90 的点云俯视 vs 官方正射裁剪。

| # | split | map | teacher step | action | pitch | valid | holes | points | render |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | val_seen | birmingham_block_11 | 92 | GoDown | -63.5° | 49.5% | 50.5% | 2,504,098 | 0.44 s |
| 1 | val_seen | birmingham_block_13 | 58 | GoDown | -45.4° | 60.0% | 40.0% | 2,350,907 | 0.29 s |
| 2 | val_seen | cambridge_block_25 | 47 | GoDown | -66.9° | 84.1% | 15.9% | 17,636,306 | 1.19 s |
| 3 | val_unseen | birmingham_block_5 | 107 | GoDown | -36.3° | 49.3% | 50.7% | 45,454,256 | 3.82 s |

每个样例的 evaluation target 都落在图像内，并记录了投影 UV 与前向深度；
birmingham_block_13 样例的 teacher marker 超出画面，其余样例的 marker 在画面内。
birmingham_block_5 的前景地面点较稀疏，出现明显空洞；这反映点云可见性/采样，
不是把俯视高度图当作 depth。

## 4. 方向与坐标校验

对 birmingham_block_11 的同一位姿做 pitch=-90 点云渲染，并与官方 top-down 正射裁剪比较；测试 4 个 yaw 偏移：

| yaw offset | 灰度相关 |
|---:|---:|
| 0° | **0.6097** |
| 90° | 0.0236 |
| 180° | 0.1860 |
| 270° | 0.0268 |

yaw=0 明显最优，说明渲染方向与官方裁剪一致；`topdown_vs_ply.jpg` 中道路标线和停车位几何对齐。

额外的单点相关性抽检（同一位姿 pitch=-90，512×512）：

| map | correlation |
|---|---:|
| birmingham_block_13 | 0.641 |
| birmingham_block_11 | 0.544 |
| birmingham_block_12 | 0.498 |
| birmingham_block_25 | 0.516 |
| birmingham_block_5 | 0.557 |
| birmingham_block_10 | 0.399 |
| birmingham_block_17 | 0.387 |
| birmingham_block_23 | 0.268 |
| cambridge_block_4 | 0.121 |

颜色相关性受点云颜色、正射影像时相、视角和空洞影响，不能作为唯一配准证据；但 PLY/TIF 的 XY 包围盒一致，yaw=0 的方向检查通过。**cambridge_block_4 等低相关地图在正式实验前需要更严格的几何配准检查**，本报告不把它们当作已验证的 AirSim 等价数据。

## 5. 当前限制

- 点云渲染没有表面重建，近地/遮挡区域会出现空洞和点状纹理；
- Depth 是相机前向深度，不是欧氏距离；
- 没有光照/材质，颜色来自原始点云；
- 相机俯仰来自方向向量，部分帧接近俯视；
- 不是官方 AirSim 渲染，不能用于声称“与官方视觉一致”；
- 只渲染了少量样例，没有生成全量 FPV 缓存。

## 6. 复现命令

```bash
python scripts/render_fpv_examples.py \
  --splits val_seen val_unseen \
  --samples 4 \
  --scan-records 5000 \
  --render-size 512 \
  --crop-radius 200 \
  --splat-radius 1 \
  --maps birmingham_block_5 birmingham_block_13 birmingham_block_11 cambridge_block_25
```

生成图片在 `figures/fpv_examples/`，完整 metadata 在 `reports/fpv_examples/metadata.json`（本地忽略）。
