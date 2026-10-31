#!/usr/bin/env python3
"""10E comparison videos: old M-v2 (blue) vs new M-v3 (orange).

Cases (from the two 5,324-task journals, real stored trajectories only):
* new win / old fail  x 2 per length group (short, medium, long)  -- the key set
* old win / new fail  x 1 per length group
* both win x 1, both fail x 1
Each stored step is held for 3 frames at 10 fps; ffmpeg x264.
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
REPO = HERE.parents[3]
assert (REPO / "experiments").is_dir(), f"REPO 解析错误: {REPO}"
RES = HERE.parent / "results"
OUT = HERE.parent / "videos"
OLD = Path("/mnt/hdd8t/Citynav-MEC/results_06/results/seed42/episodes")
sys.path.insert(0, str(REPO / "experiments/04_visual_goal_global_map_pipeline/scripts"))
from pipeline import MapRaster  # noqa: E402

W, H = 1280, 720
MAP_X1, MAP_Y1 = 900, 640
FPS, HOLD = 10, 3


def load_journal(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[r["episode_id"]] = r
    return out


def new_episode(eid: str) -> dict | None:
    for k in range(4):
        p = RES / f"10e_loop_k{k}" / "episodes" / f"{eid.replace(':', '_')}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def old_episode(eid: str) -> dict | None:
    p = OLD / f"{eid.replace(':', '_')}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pick_cases() -> list[tuple[str, str]]:
    m10e = load_journal(RES / "10e_loop" / "journal.jsonl")
    s06 = load_journal(RES / "stage06_journal.jsonl")
    ids = sorted(set(m10e) & set(s06))
    chosen: list[tuple[str, str]] = []
    used_maps: set[str] = set()

    def take(pred, label, n):
        for i in ids:
            if sum(1 for c in chosen if c[1] == label) >= n:
                return
            if i in {c[0] for c in chosen}:
                continue
            if pred(m10e[i], s06[i]) and m10e[i]["map"] not in used_maps:
                used_maps.add(m10e[i]["map"])
                chosen.append((i, label))

    for lg in ("short", "medium", "long"):
        take(lambda a, b: a["success"] and not b["success"]
             and a["length_group"] == lg, f"new win / old fail ({lg})", 2)
        take(lambda a, b: (not a["success"]) and b["success"]
             and a["length_group"] == lg, f"old win / new fail ({lg})", 1)
    take(lambda a, b: a["success"] and b["success"], "both win", 1)
    take(lambda a, b: (not a["success"]) and (not b["success"]), "both fail", 1)
    return chosen


def main() -> int:
    if not OLD.is_dir():
        raise SystemExit(f"old-M episodes missing at {OLD} (migration archive)")
    OUT.mkdir(parents=True, exist_ok=True)
    rasters: dict[str, MapRaster] = {}
    tmp_root = Path(tempfile.mkdtemp(prefix="stage10e_video_"))
    for eid, label in pick_cases():
        a = old_episode(eid)
        b = new_episode(eid)
        if a is None or b is None:
            print(f"skip {eid}: episodes missing", flush=True)
            continue
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
        x0 = max(x0, raster.left)
        x1 = min(x1, raster.right)
        y0 = max(y0, raster.bottom)
        y1 = min(y1, raster.top)
        s = min(MAP_X1 / max(x1 - x0, 1e-6), MAP_Y1 / max(y1 - y0, 1e-6))
        pw, ph = int((x1 - x0) * s), int((y1 - y0) * s)
        ox, oy = (MAP_X1 - pw) // 2, (MAP_Y1 - ph) // 2
        c0, r0 = raster.pixel(x0, y1)
        c1, r1 = raster.pixel(x1, y0)
        c0, r0 = max(int(c0), 0), max(int(r0), 0)
        c1, r1 = min(int(c1), raster.image.shape[1]), min(int(r1), raster.image.shape[0])
        base = np.zeros((H, W, 3), np.uint8)
        base[oy:oy + ph, ox:ox + pw] = cv2.resize(
            raster.image[r0:r1, c0:c1], (pw, ph), interpolation=cv2.INTER_AREA)

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
                cv2.polylines(f, [np.asarray([xy(p["x"], p["y"]) for p in bp],
                                             np.int32)],
                              False, (40, 140, 255), 2, cv2.LINE_AA)
            if len(ap) >= 2:
                cv2.polylines(f, [np.asarray([xy(p["x"], p["y"]) for p in ap],
                                             np.int32)],
                              False, (255, 140, 40), 2, cv2.LINE_AA)
            if ap:
                cv2.circle(f, xy(ap[-1]["x"], ap[-1]["y"]), 5, (255, 140, 40), -1)
            if bp:
                cv2.circle(f, xy(bp[-1]["x"], bp[-1]["y"]), 5, (40, 140, 255), -1)
            cv2.rectangle(f, (0, 650), (W, H), (25, 25, 25), -1)
            cv2.putText(f, f"{eid}  {a['split']}  {name}  |  {label}",
                        (10, 676), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(f, "OLD M-v2 (orange) step "
                           f"{min(i, len(a['trajectory']) - 1) + 1}/"
                           f"{len(a['trajectory'])} -> {a['failure']}",
                        (10, 702), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (120, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(f, "NEW M-v3 (blue) step "
                           f"{min(i, len(b['trajectory']) - 1) + 1}/"
                           f"{len(b['trajectory'])} -> {b['failure']}",
                        (560, 702), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 160, 90), 1, cv2.LINE_AA)
            for _ in range(HOLD):
                frames.append(f)
        for _ in range(HOLD * 6):
            frames.append(frames[-1])
        tmp = tmp_root / eid.replace(":", "_")
        tmp.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            cv2.imwrite(str(tmp / f"{i:05d}.png"), f)
        mp4 = OUT / f"{eid.replace(':', '_')}_oldM_vs_newM.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate",
                        str(FPS), "-i", str(tmp / "%05d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "veryfast",
                        "-movflags", "+faststart", str(mp4)], check=True)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"wrote {mp4.name} ({mp4.stat().st_size / 1e6:.2f} MB)")
    shutil.rmtree(tmp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
