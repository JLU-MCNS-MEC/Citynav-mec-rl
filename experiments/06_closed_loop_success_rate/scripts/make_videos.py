#!/usr/bin/env python3
"""Render real Stage-06 episodes into dual-view flight videos (H.264 MP4).

Left  : north-up RGB map with the 20 m success disc, start, target, the
        accumulated trajectory, the current position and heading.
Right : the actual ``heading_crop`` observation the model received at that step.
Bottom: episode id, split, length group, step, action, distance, the four
        action probabilities and the final outcome.

Nothing is re-simulated: every frame is one stored step from
``results/seed42/episodes/<id>.json``.  Each step is held for
``FRAMES_PER_STEP`` frames, so this is a discrete-action replay, not real-time
footage; the manifest records that.

    python make_videos.py --cases videos/seed42/cases.json --only val_seen:00060
    python make_videos.py --jobs 4          # all selected cases
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "04_visual_goal_global_map_pipeline/scripts"))

from pipeline import MapRaster  # noqa: E402

RES = HERE.parent / "results/seed42"
OUT = HERE.parent / "videos/seed42"

W, H = 1280, 720
MAP_X0, MAP_X1 = 0, 760
MAP_Y0, MAP_Y1 = 0, 640
OBS_X0, OBS_Y0, OBS_SIZE = 780, 0, 480
BAR_FPS = 10
FRAMES_PER_STEP = 3
CRLF = None

CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]
ACTION_COLORS = [(60, 60, 220), (200, 120, 40), (60, 170, 60), (180, 120, 200)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def out_path(case: dict) -> Path:
    sub = "success" if case["kind"] == "success" else "failure"
    return OUT / sub / case["split"] / case["length_group"] / f"{case['episode_id'].replace(':', '_')}.mp4"


def meta_path(case: dict) -> Path:
    return OUT / "meta" / f"{case['episode_id'].replace(':', '_')}.json"


def is_done(case: dict, source_sha: str) -> bool:
    mp4 = out_path(case)
    meta = meta_path(case)
    if not mp4.exists() or not meta.exists():
        return False
    try:
        m = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return m.get("source_sha256") == source_sha and m.get("bytes") == mp4.stat().st_size


def load_raster(name: str, cache: dict) -> MapRaster:
    if name not in cache:
        cache[name] = MapRaster.load(name)
    return cache[name]


def window_for(rec: dict, raster: MapRaster) -> tuple[float, float, float, float]:
    pts = [[t["x"], t["y"]] for t in rec["trajectory"]] or [rec["start"][:2]]
    pts = np.asarray(pts + [rec["target"]], np.float64)
    lo = pts.min(axis=0) - 35.0
    hi = pts.max(axis=0) + 35.0
    # enforce a minimum span so short tasks still show context
    span = np.maximum(hi - lo, 140.0)
    centre = (lo + hi) / 2
    lo, hi = centre - span / 2, centre + span / 2
    # match the panel aspect first, then shift (not crop) inside the raster
    aspect = (MAP_X1 - MAP_X0) / (MAP_Y1 - MAP_Y0)
    w, h = hi - lo
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    cx, cy = (lo + hi) / 2
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2
    if x0 < raster.left:
        x0, x1 = raster.left, raster.left + w
    if x1 > raster.right:
        x1, x0 = raster.right, raster.right - w
    if y0 < raster.bottom:
        y0, y1 = raster.bottom, raster.bottom + h
    if y1 > raster.top:
        y1, y0 = raster.top, raster.top - h
    return float(x0), float(y0), float(x1), float(y1)


class MapPanel:
    def __init__(self, raster: MapRaster, window):
        x0, y0, x1, y1 = window
        self.raster, self.x0, self.y0, self.x1, self.y1 = raster, x0, y0, x1, y1
        c0, r0 = raster.pixel(x0, y1)
        c1, r1 = raster.pixel(x1, y0)
        c0, r0 = max(int(c0), 0), max(int(r0), 0)
        c1, r1 = min(int(c1), raster.image.shape[1]), min(int(r1), raster.image.shape[0])
        self.crop = raster.image[r0:r1, c0:c1].copy()
        self.crop_origin = (raster.left + c0 * raster.res, raster.top - r0 * raster.res)
        self.s = min((MAP_X1 - MAP_X0) / max(x1 - x0, 1e-6),
                     (MAP_Y1 - MAP_Y0) / max(y1 - y0, 1e-6))
        self.pw = int((x1 - x0) * self.s)
        self.ph = int((y1 - y0) * self.s)
        self.ox = MAP_X0 + (MAP_X1 - MAP_X0 - self.pw) // 2
        self.oy = MAP_Y0 + (MAP_Y1 - MAP_Y0 - self.ph) // 2

    def xy(self, x: float, y: float) -> tuple[int, int]:
        return int(self.ox + (x - self.x0) * self.s), int(self.oy + (self.y1 - y) * self.s)

    def canvas(self) -> np.ndarray:
        panel = np.zeros((H, W, 3), np.uint8)
        img = cv2.resize(self.crop, (self.pw, self.ph), interpolation=cv2.INTER_AREA)
        panel[self.oy:self.oy + self.ph, self.ox:self.ox + self.pw] = img
        return panel


def draw_prob_bars(img: np.ndarray, probs, x: int, y: int, w: int) -> None:
    for i, (name, p) in enumerate(zip(CLASSES, probs)):
        yy = y + i * 26
        cv2.putText(img, f"{name:9s}", (x, yy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(img, (x + 95, yy + 2), (x + 95 + w, yy + 18), (60, 60, 60), -1)
        cv2.rectangle(img, (x + 95, yy + 2), (x + 95 + int(w * p), yy + 18),
                      ACTION_COLORS[i], -1)
        cv2.putText(img, f"{p:.2f}", (x + 95 + w + 6, yy + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (220, 220, 220), 1, cv2.LINE_AA)


def render_case(case: dict, rec: dict, raster: MapRaster, tmp_root: Path) -> dict:
    started = time.time()
    panel = MapPanel(raster, window_for(rec, raster))
    base = panel.canvas()
    traj = rec["trajectory"]
    target = rec["target"]
    start = rec["start"][:2]
    title = f"{rec['episode_id']}  {rec['split']}  {rec['length_group']}"
    outcome = "SUCCESS" if rec["success"] else f"FAIL: {rec['failure']}"
    if rec["stopped"] and rec["stop_dist"] is not None:
        outcome += f" | stop {rec['stop_dist']:.1f} m"
    outcome += f" | min {rec['min_dist']:.1f} m | steps {rec['steps']}"

    tx, ty = panel.xy(*target)
    cv2.circle(base, (tx, ty), int(20 * panel.s), (0, 0, 200), 1, cv2.LINE_AA)
    cv2.putText(base, "20 m", (tx + int(20 * panel.s) + 4, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 0, 200), 1, cv2.LINE_AA)
    sx, sy = panel.xy(*start)
    cv2.circle(base, (sx, sy), 6, (60, 200, 60), -1, cv2.LINE_AA)
    cv2.putText(base, "start", (sx + 8, sy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (60, 200, 60), 1, cv2.LINE_AA)
    cv2.drawMarker(base, (tx, ty), (0, 0, 255), cv2.MARKER_STAR, 18, 2)

    frames = []
    for i, st in enumerate(traj):
        frame = base.copy()
        # history trail up to the current step
        pts = np.asarray([panel.xy(t["x"], t["y"]) for t in traj[:i + 1]], np.int32)
        if len(pts) >= 2:
            cv2.polylines(frame, [pts], False, (200, 90, 30), 2, cv2.LINE_AA)
        cx, cy = panel.xy(st["x"], st["y"])
        hx = int(cx + 22 * np.cos(st["yaw"]))
        hy = int(cy - 22 * np.sin(st["yaw"]))
        cv2.arrowedLine(frame, (cx, cy), (hx, hy), (0, 165, 255), 2, cv2.LINE_AA, tipLength=0.35)
        cv2.circle(frame, (cx, cy), 4, (0, 165, 255), -1, cv2.LINE_AA)

        obs = raster.heading_crop(float(st["x"]), float(st["y"]), float(st["yaw"]))
        obs = cv2.resize(obs, (OBS_SIZE, OBS_SIZE), interpolation=cv2.INTER_NEAREST)
        frame[OBS_Y0:OBS_Y0 + OBS_SIZE, OBS_X0:OBS_X0 + OBS_SIZE] = obs
        cv2.rectangle(frame, (OBS_X0, OBS_Y0), (OBS_X0 + OBS_SIZE, OBS_Y0 + OBS_SIZE),
                      (90, 90, 90), 1)
        label = "UAV view (heading-forward, 100 m)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (OBS_X0 + 4, OBS_Y0 + 4), (OBS_X0 + 14 + tw, OBS_Y0 + 14 + th),
                      (20, 20, 20), -1)
        cv2.putText(frame, label, (OBS_X0 + 9, OBS_Y0 + 9 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        mx, my = OBS_X0 + 10, OBS_Y0 + OBS_SIZE + 18
        cv2.putText(frame, f"step {st['step'] + 1}/{rec['steps']}  action "
                           f"{CLASSES[st['action']]}", (mx, my + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"target distance {st['dist']:.1f} m", (mx, my + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        draw_prob_bars(frame, st["probs"], mx, my + 52, 150)

        cv2.rectangle(frame, (0, 650), (W, H), (25, 25, 25), -1)
        cv2.putText(frame, title, (10, 676), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"action {CLASSES[st['action']]} | dist {st['dist']:.1f} m | "
                           f"p = [{', '.join(f'{p:.2f}' for p in st['probs'])}]",
                    (10, 702), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(frame, outcome, (560, 676), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (90, 220, 90) if rec["success"] else (90, 140, 240), 1, cv2.LINE_AA)
        for _ in range(FRAMES_PER_STEP):
            frames.append(frame)

    # hold the last frame a little longer
    for _ in range(FRAMES_PER_STEP * 4):
        frames.append(frames[-1])

    tmp = Path(tmp_root) / rec["episode_id"].replace(":", "_")
    tmp.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(tmp / f"{i:05d}.png"), f)
    mp4 = out_path(case)
    mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp_mp4 = mp4.with_name(f".{mp4.name}.tmp.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(BAR_FPS),
                    "-i", str(tmp / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "26", "-preset", "veryfast", "-movflags", "+faststart",
                    str(tmp_mp4)], check=True)
    os.replace(tmp_mp4, mp4)
    shutil.rmtree(tmp, ignore_errors=True)

    cover = cv2.resize(frames[0], (640, 360), interpolation=cv2.INTER_AREA)
    cover_path = OUT / "covers" / f"{rec['episode_id'].replace(':', '_')}.png"
    cover_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cover_path), cover)
    shutil.rmtree(Path(tmp_root), ignore_errors=True)

    meta = {"episode_id": rec["episode_id"], "split": rec["split"],
            "length_group": rec["length_group"], "kind": case["kind"],
            "failure": rec["failure"], "map": rec["map"], "steps": rec["steps"],
            "fps": BAR_FPS, "frames_per_step": FRAMES_PER_STEP,
            "frames": len(frames), "duration_s": len(frames) / BAR_FPS,
            "discrete_action_replay": True,
            "source": str(RES / "episodes" / f"{rec['episode_id'].replace(':', '_')}.json"),
            "source_sha256": sha256(RES / "episodes" /
                                    f"{rec['episode_id'].replace(':', '_')}.json"),
            "mp4_sha256": sha256(mp4), "bytes": mp4.stat().st_size,
            "video": str(mp4.relative_to(OUT.parent.parent)),
            "cover": str(cover_path.relative_to(OUT.parent.parent)),
            "render_seconds": round(time.time() - started, 2),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    meta_path(case).parent.mkdir(parents=True, exist_ok=True)
    meta_path(case).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(OUT / "cases.json"))
    parser.add_argument("--only", action="append", default=[],
                        help="episode id to render (repeatable)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    payload = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = payload["cases"]
    if args.only:
        cases = [c for c in cases if c["episode_id"] in set(args.only)]
    rasters: dict[str, MapRaster] = {}
    tmp_root = Path(tempfile.mkdtemp(prefix="stage06_video_"))
    done, skipped = [], []
    for case in cases:
        src = RES / "episodes" / f"{case['episode_id'].replace(':', '_')}.json"
        rec = json.loads(src.read_text(encoding="utf-8"))
        s_sha = sha256(src)
        if not args.force and is_done(case, s_sha):
            skipped.append(case["episode_id"])
            print(f"skip {case['episode_id']} (already rendered)", flush=True)
            continue
        raster = load_raster(rec["map"], rasters)
        meta = render_case(case, rec, raster, tmp_root)
        done.append(case["episode_id"])
        print(f"rendered {case['episode_id']} {meta['duration_s']:.1f}s "
              f"{meta['bytes'] / 1e6:.2f} MB in {meta['render_seconds']}s", flush=True)
    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"done {len(done)}, skipped {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
