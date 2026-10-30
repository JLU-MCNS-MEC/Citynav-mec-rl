#!/usr/bin/env python3
"""Two paired comparison videos: Transformer-M (red) vs Top-1 geometry (blue).

Both policies start from the same pose; the trails grow together, one frame per
stored step (each held for 3 frames at 10 fps).  This is a replay of the stored
records, not a re-simulation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RES = HERE.parent / "results"
OUT = HERE.parent / "videos"
STAGE06 = ROOT / "06_closed_loop_success_rate/results/seed42"
sys.path.insert(0, str(ROOT / "04_visual_goal_global_map_pipeline/scripts"))
from pipeline import MapRaster  # noqa: E402

W, H = 1280, 720
MAP_X0, MAP_X1, MAP_Y0, MAP_Y1 = 0, 900, 0, 640
FPS, HOLD = 10, 3
CASES = [("val_unseen:00154", "T-fail / geometry-success (long)"),
         ("val_seen:00465", "T-success / geometry-fail (short)")]


def load(base: Path, eid: str) -> dict:
    return json.loads((base / f"{eid.replace(':', '_')}.json").read_text(encoding="utf-8"))


def main() -> int:
    rasters: dict[str, MapRaster] = {}
    OUT.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="stage08_video_"))
    for eid, label in CASES:
        t = load(STAGE06 / "episodes", eid)
        g = load(RES / "top1/episodes", eid)
        name = t["map"]
        raster = rasters.setdefault(name, MapRaster.load(name))
        target = np.asarray(t["target"])
        tpts = np.asarray([[s["x"], s["y"]] for s in t["trajectory"]])
        gpts = np.asarray([[s["x"], s["y"]] for s in g["trajectory"]])
        allp = np.vstack([tpts, gpts, target[None]])
        lo = allp.min(axis=0) - 30
        hi = allp.max(axis=0) + 30
        w, h = np.maximum(hi - lo, 120.0)
        aspect = (MAP_X1 - MAP_X0) / (MAP_Y1 - MAP_Y0)
        if w / h < aspect:
            w = h * aspect
        else:
            h = w / aspect
        cx, cy = (lo + hi) / 2
        x0, x1 = cx - w / 2, cx + w / 2
        y0, y1 = cy - h / 2, cy + h / 2
        x0 = max(x0, raster.left); x1 = min(x1, raster.right)
        y0 = max(y0, raster.bottom); y1 = min(y1, raster.top)
        s = min((MAP_X1 - MAP_X0) / max(x1 - x0, 1e-6), (MAP_Y1 - MAP_Y0) / max(y1 - y0, 1e-6))
        pw, ph = int((x1 - x0) * s), int((y1 - y0) * s)
        ox, oy = (MAP_X1 - pw) // 2, (MAP_Y1 - ph) // 2
        c0, r0 = raster.pixel(x0, y1)
        c1, r1 = raster.pixel(x1, y0)
        c0, r0 = max(int(c0), 0), max(int(r0), 0)
        c1, r1 = min(int(c1), raster.image.shape[1]), min(int(r1), raster.image.shape[0])
        base = np.zeros((H, W, 3), np.uint8)
        base[oy:oy + ph, ox:ox + pw] = cv2.resize(raster.image[r0:r1, c0:c1], (pw, ph),
                                                  interpolation=cv2.INTER_AREA)

        def xy(x, y):
            return int(ox + (x - x0) * s), int(oy + (y1 - y) * s)

        tx, ty = xy(*target)
        cv2.circle(base, (tx, ty), int(20 * s), (0, 0, 200), 1, cv2.LINE_AA)
        cv2.drawMarker(base, (tx, ty), (0, 0, 255), cv2.MARKER_STAR, 18, 2)
        n = max(len(t["trajectory"]), len(g["trajectory"]))
        frames = []
        for i in range(n):
            f = base.copy()
            tp = t["trajectory"][:min(i + 1, len(t["trajectory"]))]
            gp = g["trajectory"][:min(i + 1, len(g["trajectory"]))]
            if len(tp) >= 2:
                cv2.polylines(f, [np.asarray([xy(p["x"], p["y"]) for p in tp], np.int32)],
                              False, (60, 60, 210), 2, cv2.LINE_AA)
            if len(gp) >= 2:
                cv2.polylines(f, [np.asarray([xy(p["x"], p["y"]) for p in gp], np.int32)],
                              False, (200, 90, 30), 2, cv2.LINE_AA)
            if tp:
                p = tp[-1]
                cv2.circle(f, xy(p["x"], p["y"]), 5, (60, 60, 210), -1)
                cv2.arrowedLine(f, xy(p["x"], p["y"]),
                                xy(p["x"] + 20 * np.cos(p["yaw"]), p["y"] + 20 * np.sin(p["yaw"])),
                                (60, 60, 210), 1, cv2.LINE_AA, tipLength=0.4)
            if gp:
                p = gp[-1]
                cv2.circle(f, xy(p["x"], p["y"]), 5, (200, 90, 30), -1)

            cv2.rectangle(f, (0, 650), (W, H), (25, 25, 25), -1)
            cv2.putText(f, f"{eid}  {t['split']}  {t['map']}  |  {label}",
                        (10, 676), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            t_state = (f"T step {min(i, len(t['trajectory']) - 1) + 1}/{len(t['trajectory'])}"
                       f" -> {t['failure']}")
            g_state = (f"G step {min(i, len(g['trajectory']) - 1) + 1}/{len(g['trajectory'])}"
                       f" -> {g['failure']}")
            cv2.putText(f, t_state, (10, 702), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (90, 90, 235), 1, cv2.LINE_AA)
            cv2.putText(f, g_state, (520, 702), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (235, 140, 60), 1, cv2.LINE_AA)
            for _ in range(HOLD):
                frames.append(f)
        for _ in range(HOLD * 6):
            frames.append(frames[-1])

        tmp = tmp_root / eid.replace(":", "_")
        tmp.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            cv2.imwrite(str(tmp / f"{i:05d}.png"), f)
        mp4 = OUT / f"{eid.replace(':', '_')}_compare.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", str(tmp / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "26", "-preset", "veryfast", "-movflags", "+faststart", str(mp4)],
                       check=True)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"wrote {mp4} ({mp4.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")
    shutil.rmtree(tmp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
