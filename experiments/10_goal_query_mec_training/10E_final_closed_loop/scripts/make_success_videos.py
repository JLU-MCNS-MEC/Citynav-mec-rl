#!/usr/bin/env python3
"""Dual-view navigation replay videos for the five selected 10E episodes.

Layout follows Stage-06 ``make_videos.py`` (1280x720, 16:9):
  left  0..760 x 0..640   north-up RGB map: 20 m success disc, start, target,
                          accumulated trajectory, current position + heading
  right 780..1260 x 0..480 heading-forward 100 m observation (heading_crop),
                          labelled "UAV view (heading-forward, 100 m)"
  right-bottom            step, action, target distance + four softmax bars
  bottom bar              episode id / split / length group, action + distance
                          + probabilities, final outcome (stop/min/steps)

Probabilities come from the offline replay (replay_10e.py, real best.pt
forward passes -- never one-hot); the displayed target distance is the stored
``true_dist_m`` and is NOT a model input.  Each stored step is held for 3
frames at 10 fps; the last frame is held longer.  Per-episode rendering is
resumable (source + replay SHA keyed).

    python make_success_videos.py --only val_seen:00018
    python make_success_videos.py            # all five
    python make_success_videos.py --compile  # concatenate compilations
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
REPO = HERE.parents[3]
assert (REPO / "experiments").is_dir(), f"REPO 解析错误: {REPO}"
sys.path.insert(0, str(REPO / "experiments/04_visual_goal_global_map_pipeline/scripts"))
from pipeline import MapRaster  # noqa: E402

E = HERE.parent.parent / "10E_final_closed_loop"
RES = E / "results"
OUT = E / "videos" / "success_examples_5"
REPLAY = OUT / "replay"

W, H = 1280, 720
MAP_X0, MAP_X1 = 0, 760
MAP_Y0, MAP_Y1 = 0, 640
OBS_X0, OBS_Y0, OBS_SIZE = 780, 0, 480
BAR_FPS = 10
FRAMES_PER_STEP = 3
CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]
ACTION_COLORS = [(60, 60, 220), (200, 120, 40), (60, 170, 60), (180, 120, 200)]

# the five requested episodes, in order; 00960 is the new-M FAILURE contrast
CASES = [("val_seen:00018", "new M-v3 success / old M-v2 fail"),
         ("val_seen:00591", "new M-v3 success / old M-v2 fail"),
         ("val_seen:00629", "new M-v3 success / old M-v2 fail"),
         ("val_seen:00776", "new M-v3 success / old M-v2 fail"),
         ("val_seen:00960", "new M-v3 fail / old M-v2 success (contrast)")]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_episode(eid: str) -> tuple[dict, Path]:
    for k in range(4):
        p = RES / f"10e_loop_k{k}" / "episodes" / f"{eid.replace(':', '_')}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")), p
    raise SystemExit(f"episode {eid} not found")


def window_for(rec: dict, raster: MapRaster) -> tuple[float, float, float, float]:
    pts = [[t["x"], t["y"]] for t in rec["trajectory"]] or [rec["start"][:2]]
    pts = np.asarray(pts + [rec["target"]], np.float64)
    lo = pts.min(axis=0) - 35.0
    hi = pts.max(axis=0) + 35.0
    span = np.maximum(hi - lo, 140.0)
    centre = (lo + hi) / 2
    lo, hi = centre - span / 2, centre + span / 2
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
        cv2.putText(img, f"{p:.2f}", (x + 95 + w + 6, yy + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_00 if False else cv2.LINE_AA)


def render_case(eid: str, label: str, raster: MapRaster, tmp_root: Path) -> dict:
    started = time.time()
    rec, src = load_episode(eid)
    rep = json.loads((REPLAY / f"{eid.replace(':', '_')}.json").read_text(
        encoding="utf-8"))
    assert rep["argmax_fully_consistent"], f"{eid}: replay argmax inconsistent"
    probs_by_step = {s["step"]: s["probs"] for s in rep["trajectory"]}

    panel = MapPanel(raster, window_for(rec, raster))
    base = panel.canvas()
    traj = rec["trajectory"]
    target = rec["target"]
    start = rec["start"][:2]
    title = f"{rec['episode_id']}  {rec['split']}  {rec['length_group']}  |  {label}"
    outcome = "SUCCESS" if rec["success"] else f"FAIL: {rec['failure']}"
    if rec["stopped"] and rec.get("stop_dist") is not None:
        outcome += f" | stop {rec['stop_dist']:.1f} m"
    outcome += f" | min {rec['min_dist']:.1f} m | steps {rec['steps']}"

    tx, ty = panel.xy(*target)
    cv2.circle(base, (tx, ty), int(20 * panel.s), (0, 0, 200), 1, cv2.LINE_AA)
    cv2.putText(base, "20 m", (tx + int(20 * panel.s) + 4, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1, cv2.LINE_AA)
    sx, sy = panel.xy(*start)
    cv2.circle(base, (sx, sy), 6, (60, 200, 60), -1, cv2.LINE_AA)
    cv2.putText(base, "start", (sx + 8, sy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (60, 200, 60), 1, cv2.LINE_AA)
    cv2.drawMarker(base, (tx, ty), (0, 0, 255), cv2.MARKER_STAR, 18, 2)

    frames = []
    for i, st in enumerate(traj):
        frame = base.copy()
        pts = np.asarray([panel.xy(t["x"], t["y"]) for t in traj[:i + 1]], np.int32)
        if len(pts) >= 2:
            cv2.polylines(frame, [pts], False, (200, 90, 30), 2, cv2.LINE_AA)
        cx, cy = panel.xy(st["x"], st["y"])
        hx = int(cx + 22 * np.cos(st["yaw"]))
        hy = int(cy - 22 * np.sin(st["yaw"]))
        cv2.arrowedLine(frame, (cx, cy), (hx, hy), (0, 165, 255), 2, cv2.LINE_AA,
                        tipLength=0.35)
        cv2.circle(frame, (cx, cy), 4, (0, 165, 255), -1, cv2.LINE_AA)

        obs = raster.heading_crop(float(st["x"]), float(st["y"]), float(st["yaw"]))
        obs = cv2.resize(obs, (OBS_SIZE, OBS_SIZE), interpolation=cv2.INTER_NEAREST)
        frame[OBS_Y0:OBS_Y0 + OBS_SIZE, OBS_X0:OBS_X0 + OBS_SIZE] = obs
        cv2.rectangle(frame, (OBS_X0, OBS_Y0), (OBS_X0 + OBS_SIZE, OBS_Y0 + OBS_SIZE),
                      (90, 90, 90), 1)
        lab = "UAV view (heading-forward, 100 m)"
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (OBS_X0 + 4, OBS_Y0 + 4),
                      (OBS_X0 + 14 + tw, OBS_Y0 + 14 + th), (20, 20, 20), -1)
        cv2.putText(frame, lab, (OBS_X0 + 9, OBS_Y0 + 9 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        probs = probs_by_step[st["step"]]
        dist = float(st["true_dist_m"])
        mx, my = OBS_X0 + 10, OBS_Y0 + OBS_SIZE + 18
        cv2.putText(frame, f"step {st['step'] + 1}/{rec['steps']}  action "
                           f"{CLASSES[st['action']]}", (mx, my + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"target distance {dist:.1f} m", (mx, my + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        draw_prob_bars(frame, probs, mx, my + 52, 150)

        cv2.rectangle(frame, (0, 650), (W, H), (25, 25, 25), -1)
        cv2.putText(frame, title, (10, 676), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"action {CLASSES[st['action']]} | dist {dist:.1f} m | "
                           f"p = [{', '.join(f'{p:.2f}' for p in probs)}]",
                    (10, 702), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, outcome, (480, 676), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (90, 220, 90) if rec["success"] else (90, 140, 240), 1,
                    cv2.LINE_AA)
        for _ in range(FRAMES_PER_STEP):
            frames.append(frame)
    for _ in range(FRAMES_PER_STEP * 4):
        frames.append(frames[-1])

    tmp = Path(tmp_root) / eid.replace(":", "_")
    tmp.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(tmp / f"{i:05d}.png"), f)
    ind = OUT / "individual"
    ind.mkdir(parents=True, exist_ok=True)
    mp4 = ind / f"{eid.replace(':', '_')}.mp4"
    tmp_mp4 = mp4.with_name(f".{mp4.name}.tmp.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(BAR_FPS),
                    "-i", str(tmp / "%05d.png"), "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-crf", "26", "-preset", "veryfast",
                    "-movflags", "+faststart", str(tmp_mp4)], check=True)
    os.replace(tmp_mp4, mp4)
    shutil.rmtree(tmp, ignore_errors=True)

    covers = OUT / "covers"
    covers.mkdir(parents=True, exist_ok=True)
    cover = covers / f"{eid.replace(':', '_')}.png"
    cv2.imwrite(str(cover), cv2.resize(frames[0], (640, 360),
                                       interpolation=cv2.INTER_AREA))

    meta = {"episode_id": eid, "split": rec["split"],
            "length_group": rec["length_group"], "label": label,
            "success": rec["success"], "failure": rec["failure"],
            "map": rec["map"], "steps": rec["steps"],
            "fps": BAR_FPS, "frames_per_step": FRAMES_PER_STEP,
            "frames": len(frames), "duration_s": len(frames) / BAR_FPS,
            "discrete_action_replay": True,
            "source": str(src), "source_sha256": sha256(src),
            "replay": str(REPLAY / f"{eid.replace(':', '_')}.json"),
            "replay_argmax_consistent": rep["argmax_fully_consistent"],
            "checkpoint_sha256": rep["checkpoint_sha256"],
            "mp4_sha256": sha256(mp4), "bytes": mp4.stat().st_size,
            "render_seconds": round(time.time() - started, 2),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (OUT / "meta" / f"{eid.replace(':', '_')}.json").parent.mkdir(
        parents=True, exist_ok=True)
    (OUT / "meta" / f"{eid.replace(':', '_')}.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def compile_videos(ids: list[str], out_name: str, title: str) -> Path:
    comp_dir = OUT / "compilations"
    comp_dir.mkdir(parents=True, exist_ok=True)
    list_file = comp_dir / f".{out_name}.list"
    with list_file.open("w", encoding="utf-8") as fh:
        for eid in ids:
            mp4 = OUT / "individual" / f"{eid.replace(':', '_')}.mp4"
            fh.write(f"file '{mp4}'\n")
    out = comp_dir / out_name
    tmp = out.with_name(f".{out.name}.tmp.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(list_file), "-c", "copy", "-movflags", "+faststart",
                    str(tmp)], check=True)
    os.replace(tmp, out)
    list_file.unlink(missing_ok=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", default=[],
                        help="episode id (repeatable; default: the 5-case set)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("CITYNAV_DATA"):
        raise SystemExit("CITYNAV_DATA is not set")
    cases = [c for c in CASES if not args.only or c[0] in set(args.only)]
    if not args.compile:
        rasters: dict[str, MapRaster] = {}
        tmp_root = Path(tempfile.mkdtemp(prefix="stage10e_success_video_"))
        for eid, label in cases:
            rec, src = load_episode(eid)
            rep_path = REPLAY / f"{eid.replace(':', '_')}.json"
            s_sha = sha256(src) + sha256(rep_path)
            meta_path = OUT / "meta" / f"{eid.replace(':', '_')}.json"
            if not args.force and meta_path.exists() and \
                    (OUT / "individual" / f"{eid.replace(':', '_')}.mp4").exists():
                m = json.loads(meta_path.read_text(encoding="utf-8"))
                if m.get("source_sha256") == sha256(src):
                    print(f"skip {eid} (rendered)", flush=True)
                    continue
            raster = rasters.setdefault(rec["map"], MapRaster.load(rec["map"]))
            meta = render_case(eid, label, raster, tmp_root)
            print(f"rendered {eid} {meta['duration_s']:.1f}s "
                  f"{meta['bytes'] / 1e6:.2f} MB | success={meta['success']} | "
                  f"argmax_consistent={meta['replay_argmax_consistent']}", flush=True)
        shutil.rmtree(tmp_root, ignore_errors=True)
    compile_videos(["val_seen:00018", "val_seen:00591", "val_seen:00629",
                    "val_seen:00776"], "m_v3_success_compilation_4.mp4",
                   "new M-v3 success x4")
    compile_videos([c[0] for c in CASES], "selected_5_cases_compilation.mp4",
                   "selected 5 cases (00960 = contrast failure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
