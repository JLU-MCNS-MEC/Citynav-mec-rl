#!/usr/bin/env python3
"""Four paired comparison videos: Continuous C1 (blue) vs Static Top-1 (red).

Real stored records only; each stored step is held for 3 frames at 10 fps.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RES = HERE.parent / "results"
OUT = HERE.parent / "videos"
C1 = RES / "09c_c1" / "episodes"
S8 = ROOT / "experiments/08_belief_only_geometric_navigation/results/top1/episodes"
sys.path.insert(0, str(ROOT / "experiments/04_visual_goal_global_map_pipeline/scripts"))
from pipeline import MapRaster  # noqa: E402

W, H = 1280, 720
MAP_X1, MAP_Y1 = 900, 640
FPS, HOLD = 10, 3


def load(base: Path, eid: str) -> dict:
    return json.loads((base / f"{eid.replace(':', '_')}.json").read_text(encoding="utf-8"))


def pick_cases() -> list[tuple[str, str]]:
    c1 = {json.loads(p.read_text(encoding="utf-8"))["episode_id"]:
          json.loads(p.read_text(encoding="utf-8")) for p in sorted(C1.glob("*.json"))}
    s8 = {json.loads(p.read_text(encoding="utf-8"))["episode_id"]:
          json.loads(p.read_text(encoding="utf-8")) for p in sorted(S8.glob("*.json"))}
    ids = sorted(set(c1) & set(s8))
    chosen, used_maps = [], set()

    def take(pred, label, n):
        for i in ids:
            if len([c for c in chosen if c[1].startswith(label)]) >= n:
                break
            if i in {c[0] for c in chosen}:
                continue
            if pred(c1[i], s8[i]) and c1[i]["map"] not in used_maps:
                used_maps.add(c1[i]["map"])
                chosen.append((i, label))

    take(lambda a, b: a["success"] and not b["success"] and a["length_group"] == "long",
         "C1 win, long", 2)
    take(lambda a, b: a["success"] and not b["success"], "C1 win", 2)
    take(lambda a, b: (not a["success"]) and b["success"], "Top-1 win", 1)
    take(lambda a, b: (not a["success"]) and (not b["success"]), "both fail", 1)
    return chosen[:4]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rasters: dict[str, MapRaster] = {}
    tmp_root = Path(tempfile.mkdtemp(prefix="stage09c_video_"))
    for eid, label in pick_cases():
        a = load(C1, eid)
        b = load(S8, eid)
        name = a["map"]
        raster = rasters.setdefault(name, MapRaster.load(name))
        target = np.asarray(a["target"])
        apts = np.asarray([[s["x"], s["y"]] for s in a["trajectory"]])
        bpts = np.asarray([[s["x"], s["y"]] for s in b["trajectory"]])
        allp = np.vstack([apts, bpts, target[None]])
        lo = allp.min(axis=0) - 30
        hi = allp.max(axis=0) + 30
        w, h = np.maximum(hi - lo, 120.0)
        aspect = MAP_X1 / MAP_Y1
        if w / h < aspect:
            w = h * aspect
        else:
            h = w / aspect
        cx, cy = (lo + hi) / 2
        x0, x1 = cx - w / 2, cx + w / 2
        y0, y1 = cy - h / 2, cy + h / 2
        x0 = max(x0, raster.left); x1 = min(x1, raster.right)
        y0 = max(y0, raster.bottom); y1 = min(y1, raster.top)
        s = min(MAP_X1 / max(x1 - x0, 1e-6), MAP_Y1 / max(y1 - y0, 1e-6))
        pw, ph = int((x1 - x0) * s), int((y1 - y0) * s)
        ox, oy = (MAP_X1 - pw) // 2, (MAP_Y1 - ph) // 2
        c0, r0 = raster.pixel(x0, y1)
        c1_, r1 = raster.pixel(x1, y0)
        c0, r0 = max(int(c0), 0), max(int(r0), 0)
        c1_, r1 = min(int(c1_), raster.image.shape[1]), min(int(r1), raster.image.shape[0])
        base = np.zeros((H, W, 3), np.uint8)
        base[oy:oy + ph, ox:ox + pw] = cv2.resize(raster.image[r0:r1, c0:c1_], (pw, ph),
                                                  interpolation=cv2.INTER_AREA)

        def xy(x, y):
            return int(ox + (x - x0) * s), int(oy + (y1 - y) * s)

        tx, ty = xy(*target)
        cv2.circle(base, (tx, ty), int(20 * s), (0, 0, 200), 1, cv2.LINE_AA)
        cv2.drawMarker(base, (tx, ty), (0, 0, 255), cv2.MARKER_STAR, 18, 2)
        n = max(len(a["trajectory"]), len(b["trajectory"]))
        frames = []
        for i in range(n):
            f = base.copy()
            ap = a["trajectory"][:min(i + 1, len(a["trajectory"]))]
            bp = b["trajectory"][:min(i + 1, len(b["trajectory"]))]
            if len(bp) >= 2:
                cv2.polylines(f, [np.asarray([xy(p["x"], p["y"]) for p in bp], np.int32)],
                              False, (60, 60, 210), 2, cv2.LINE_AA)
            if len(ap) >= 2:
                cv2.polylines(f, [np.asarray([xy(p["x"], p["y"]) for p in ap], np.int32)],
                              False, (200, 90, 30), 2, cv2.LINE_AA)
            if ap:
                p = ap[-1]
                cv2.circle(f, xy(p["x"], p["y"]), 5, (200, 90, 30), -1)
                est = p.get("estimate")
                if est is not None:
                    cv2.drawMarker(f, xy(est[0], est[1]), (0, 220, 220),
                                   cv2.MARKER_TILTED_CROSS, 12, 2)
            if bp:
                p = bp[-1]
                cv2.circle(f, xy(p["x"], p["y"]), 5, (60, 60, 210), -1)
            cv2.rectangle(f, (0, 650), (W, H), (25, 25, 25), -1)
            cv2.putText(f, f"{eid}  {a['split']}  {name}  |  {label}", (10, 676),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(f, f"C1 step {min(i, len(a['trajectory']) - 1) + 1}/"
                           f"{len(a['trajectory'])} -> {a['failure']}", (10, 702),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 140, 60), 1, cv2.LINE_AA)
            cv2.putText(f, f"Top-1 step {min(i, len(b['trajectory']) - 1) + 1}/"
                           f"{len(b['trajectory'])} -> {b['failure']}", (560, 702),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 235), 1, cv2.LINE_AA)
            for _ in range(HOLD):
                frames.append(f)
        for _ in range(HOLD * 6):
            frames.append(frames[-1])
        tmp = tmp_root / eid.replace(":", "_")
        tmp.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            cv2.imwrite(str(tmp / f"{i:05d}.png"), f)
        mp4 = OUT / f"{eid.replace(':', '_')}_c1_vs_top1.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", str(tmp / "%05d.png"), "-c:v", "libx264", "-pix_fmt",
                        "yuv420p", "-crf", "26", "-preset", "veryfast",
                        "-movflags", "+faststart", str(mp4)], check=True)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"wrote {mp4} ({mp4.stat().st_size / 1e6:.2f} MB)")
    shutil.rmtree(tmp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
