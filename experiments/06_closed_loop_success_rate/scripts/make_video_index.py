#!/usr/bin/env python3
"""Build VIDEO_INDEX.md, video_manifest.csv, VIDEO_REPORT.md and the montage.

Also concatenates the 12 success clips into success_compilation.mp4 (identical
encode settings, stream copy) so the summary video contains only successes.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "videos/seed42"


def main() -> int:
    cases = json.loads((OUT / "cases.json").read_text(encoding="utf-8"))["cases"]
    metas = {}
    for c in cases:
        p = OUT / "meta" / f"{c['episode_id'].replace(':', '_')}.json"
        if p.exists():
            metas[c["episode_id"]] = json.loads(p.read_text(encoding="utf-8"))

    rows = []
    for c in cases:
        m = metas.get(c["episode_id"])
        if not m:
            continue
        rows.append({"episode_id": c["episode_id"], "kind": c["kind"],
                     "label": c.get("label", "success"), "split": c["split"],
                     "length_group": c["length_group"], "map": c["map"],
                     "steps": m["steps"], "duration_s": m["duration_s"],
                     "fps": m["fps"], "frames_per_step": m["frames_per_step"],
                     "bytes": m["bytes"], "sha256": m["mp4_sha256"],
                     "video": m["video"], "cover": m["cover"]})
    with (OUT / "video_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # montage of covers (4 x 5)
    covers = [cv2.imread(str(HERE.parent / r["cover"])) for r in rows]
    covers = [c for c in covers if c is not None]
    cols, tile_w, tile_h = 4, 480, 270
    rows_n = (len(covers) + cols - 1) // cols
    canvas = np.zeros((rows_n * tile_h, cols * tile_w, 3), np.uint8)
    for i, c in enumerate(covers):
        r, q = divmod(i, cols)
        canvas[r * tile_h:(r + 1) * tile_h, q * tile_w:(q + 1) * tile_w] = cv2.resize(
            c, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / "video_montage.png"), canvas)

    # success compilation
    succ = [r for r in rows if r["kind"] == "success"]
    listing = OUT / "success_concat.txt"
    with listing.open("w", encoding="utf-8") as fh:
        for r in succ:
            fh.write(f"file '{HERE.parent / r['video']}'\n")
    comp = OUT / "success_compilation.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(comp)],
                   check=True)
    listing.unlink(missing_ok=True)

    total_bytes = sum(r["bytes"] for r in rows) + comp.stat().st_size

    # VIDEO_INDEX.md
    L = ["# Stage 06 闭环飞行视频索引", "",
         f"共 **{len(rows)}** 条真实闭环 episode："
         f"{sum(1 for r in rows if r['kind'] == 'success')} 条成功、"
         f"{sum(1 for r in rows if r['kind'] == 'failure')} 条失败诊断对照。",
         "", f"全部视频合计 {total_bytes / 1e6:.1f} MB。", "",
         "> 视频是**离散动作回放**：每步 3 帧、10 fps，按住真实状态与动作顺序播放，"
         "不是真实物理时间录像。视觉帧来自 Stage 06 的 `heading_crop()`。", "",
         "## 成功案例", ""]
    for split in ("val_seen", "val_unseen"):
        for group in ("short", "medium", "long"):
            sel = [r for r in rows if r["kind"] == "success" and r["split"] == split
                   and r["length_group"] == group]
            if not sel:
                continue
            L.append(f"### {split} / {group}")
            L.append("")
            for r in sel:
                L.append(f"- `{r['episode_id']}` ({r['map']}, {r['steps']} steps, "
                         f"{r['duration_s']:.1f}s): [MP4]({r['video']}) · "
                         f"[封面]({r['cover']})")
            L.append("")
    L += ["## 失败诊断案例", ""]
    for r in rows:
        if r["kind"] == "failure":
            L.append(f"- `{r['episode_id']}` — {r['label']} ({r['split']}/{r['length_group']}, "
                     f"{r['map']}, {r['steps']} steps): [MP4]({r['video']}) · "
                     f"[封面]({r['cover']})")
    L += ["", "## 汇总", "",
          f"- 12 条成功轨迹汇总视频：[success_compilation.mp4](success_compilation.mp4) "
          f"({comp.stat().st_size / 1e6:.1f} MB)",
          "- 封面拼图：[video_montage.png](video_montage.png)",
          "- 清单：[video_manifest.csv](video_manifest.csv)（含每个 MP4 的 SHA-256）", ""]
    (OUT / "VIDEO_INDEX.md").write_text("\n".join(L), encoding="utf-8")

    # VIDEO_REPORT.md
    R = ["# Stage 06 闭环飞行视频报告", "",
         "## 1. 目的", "",
         "把 Stage 06 保存的真实闭环轨迹渲染成可观看视频，用于人工检查模型行为："
         "是否朝目标飞、何时转向、何时 STOP、失败发生在什么位置。", "",
         "**没有重新运行模型、没有重新生成轨迹、没有修改任何动作或概率。**"
         "每帧都来自 `results/seed42/episodes/<id>.json` 的原始逐步记录。", "",
         "## 2. 生成方式", "",
         "- 左：North-up 真实 RGB 地图（数据 `DATA/rgbd/`），绘制 20 m 成功圈、起点、目标、"
         "实际轨迹、当前位置与航向。",
         "- 右：当前位姿的 `MapRaster.heading_crop()` 局部俯视 224×224（放大 2 倍），"
         "与模型输入完全一致。",
         "- 底部：episode id、划分、长度组、step、动作、目标距离、四类概率、最终结果。",
         "- 编码：H.264 (`libx264`)，1280×720，10 fps，crf 26；每步保持 3 帧。", "",
         "## 3. 案例选择", "",
         "成功案例：每个（划分 × 长度组）2 条，优先选择地图不同、转弯次数较多的任务；"
         "失败案例：按失败类型各 1 条，包括过早 STOP、进入 20 m 未停、长任务零转向越界、"
         "转向振荡、低有效目标图像和碰撞。", "",
         "| episode | 类型 | 划分 | 长度 | 地图 | 步数 | 时长 | 文件 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        R.append(f"| {r['episode_id']} | {r['kind']}{'/' + r['label'] if r['kind'] == 'failure' else ''} "
                 f"| {r['split']} | {r['length_group']} | {r['map']} | {r['steps']} | "
                 f"{r['duration_s']:.1f}s | `{Path(r['video']).name}` |")
    R += ["", "## 4. 体量与许可", "",
          f"- 全部 MP4 合计约 {total_bytes / 1e6:.1f} MB，最大单文件 "
          f"{max(r['bytes'] for r in rows) / 1e6:.1f} MB，低于 GitHub 100 MB 限制。",
          "- 视频由本项目自己的地图数据与模型轨迹渲染，不包含第三方版权素材。", "",
          "## 5. 限制", "",
          "- 离散动作回放，不是实时飞行录像；帧数 = 步数 × 3 + 尾部停留。",
          "- 局部视图是 `heading_crop` 的静态透视裁剪，不含渲染/光照。",
          "- 视频只用于人工定性检查；定量结论以 Stage 06 的指标表为准。", ""]
    (OUT / "VIDEO_REPORT.md").write_text("\n".join(R), encoding="utf-8")

    print(f"wrote {OUT / 'VIDEO_INDEX.md'}, VIDEO_REPORT.md, video_manifest.csv, "
          f"montage, compilation ({comp.stat().st_size / 1e6:.1f} MB); total {total_bytes / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
