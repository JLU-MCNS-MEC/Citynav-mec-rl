#!/usr/bin/env python3
"""Comparison videos: two models on the same episode, time-synchronised.

1280x720 / 16:9 / H.264 layout (per the case-study spec):
  left            shared top-down map (640x640): 20 m target disc + star,
                  start marker, per-model accumulated trajectories (model
                  colours), per-model current pose + heading arrow, per-model
                  belief estimate (TILTED_CROSS, from the stored trajectory
                  `estimate` -- never fabricated), per-model final position
                  marker, legend
  right-top       per-model heading-forward 100 m observation (288x288 each)
                  with an ENDED badge once a model's trajectory is over
                  (frozen final state, clearly marked -- never extended)
  right-bottom    per-model four-action softmax bars (real checkpoint
                  forwards from replay_corrected.py -- never one-hot)
  bottom bar      episode id / split / map / difficulty, per-model result
                  (steps, final/min/stop distances, failure reason), and the
                  note "target marker: visualization only — not policy input"

Time sync: frames advance over max(len(trajA), len(trajB)) steps; a model
whose trajectory ended keeps its final state, frozen.

    python make_comparison_videos.py --selection case_studies/selection.json
    python make_comparison_videos.py --category A --only val_unseen:04231
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
REPO = HERE.parents[2]
assert (REPO / "experiments").is_dir(), f"REPO 解析错误: {REPO}"
sys.path.insert(0, str(REPO / "experiments/04_visual_goal_global_map_pipeline"
                        / "scripts"))
from pipeline import MapRaster  # noqa: E402

CASE_ROOT = REPO / "reports/v4_v5_comparison/case_studies"
SELECTION = CASE_ROOT / "selection.json"
REPLAY_DIR = CASE_ROOT / "metadata" / "replay"
META_DIR = CASE_ROOT / "metadata"
GEN_DIR = META_DIR / "generation"
COVERS = CASE_ROOT / "covers"
METRICS_V4V5 = REPO / "reports/v4_v5_comparison/metrics.json"
METRICS_V3 = REPO / "reports/v4_v5_comparison/v3_revalidation_metrics.json"

W, H = 1280, 720
MAP_X0, MAP_Y0, MAP_X1, MAP_Y1 = 0, 0, 640, 640
OBS_SIZE = 288
OBS_SLOTS = {"winner": (648, 0), "loser": (952, 0)}
BAR_SLOTS = {"winner": (648, 300), "loser": (952, 300)}
BAR_W = 150
BAR_LABEL_W = 78
FPS, HOLD, TAIL = 10, 3, 6
CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]
ACTION_COLORS = [(60, 60, 220), (200, 120, 40), (60, 170, 60), (180, 120, 200)]
MODEL_COLORS = {"v4": (200, 90, 30), "v5": (60, 120, 255), "v3": (60, 200, 60)}
STOP_RADIUS = 20.0


# --------------------------------------------------------------- pure helpers


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def window_for(pts_lists: list, target, bounds: dict,
               panel_w: float, panel_h: float,
               min_span: float = 140.0, pad: float = 35.0) -> tuple:
    all_pts = [p for pts in pts_lists for p in pts]
    pts = np.asarray(all_pts + [target], np.float64)
    lo = pts.min(axis=0) - pad
    hi = pts.max(axis=0) + pad
    span = np.maximum(hi - lo, min_span)
    centre = (lo + hi) / 2
    lo, hi = centre - span / 2, centre + span / 2
    aspect = panel_w / panel_h
    w, hgt = hi - lo
    if w / hgt < aspect:
        w = hgt * aspect
    else:
        hgt = w / aspect
    cx, cy = (lo + hi) / 2
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - hgt / 2, cy + hgt / 2
    if x0 < bounds["left"]:
        x0, x1 = bounds["left"], bounds["left"] + w
    if x1 > bounds["right"]:
        x1, x0 = bounds["right"], bounds["right"] - w
    if y0 < bounds["bottom"]:
        y0, y1 = bounds["bottom"], bounds["bottom"] + hgt
    if y1 > bounds["top"]:
        y1, y0 = bounds["top"], bounds["top"] - hgt
    return float(x0), float(y0), float(x1), float(y1)


class MapPanel:
    """World-window -> 640x640 pixel panel (north-up), decoupled from
    MapRaster so the geometry is unit-testable with synthetic crops."""

    def __init__(self, crop: np.ndarray, x0: float, y0: float, x1: float,
                 y1: float, panel_w: int = MAP_X1 - MAP_X0,
                 panel_h: int = MAP_Y1 - MAP_Y0):
        self.crop = crop
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.s = min(panel_w / max(x1 - x0, 1e-6),
                     panel_h / max(y1 - y0, 1e-6))
        self.pw = int((x1 - x0) * self.s)
        self.ph = int((y1 - y0) * self.s)
        self.ox = MAP_X0 + (panel_w - self.pw) // 2
        self.oy = MAP_Y0 + (panel_h - self.ph) // 2

    def xy(self, x: float, y: float) -> tuple[int, int]:
        return (int(self.ox + (x - self.x0) * self.s),
                int(self.oy + (self.y1 - y) * self.s))

    def canvas(self) -> np.ndarray:
        panel = np.zeros((H, W, 3), np.uint8)
        img = cv2.resize(self.crop, (self.pw, self.ph),
                         interpolation=cv2.INTER_AREA)
        panel[self.oy:self.oy + self.ph, self.ox:self.ox + self.pw] = img
        return panel


def step_for(i: int, n: int) -> tuple[int, bool]:
    """Sync index: clamp to the last step; ended once the frame index is
    at or past the trajectory length."""
    idx = min(i, n - 1)
    return idx, i >= n


def draw_prob_bars(img: np.ndarray, probs, x: int, y: int, w: int = BAR_W) \
        -> None:
    for i, (name, p) in enumerate(zip(CLASSES, probs)):
        yy = y + i * 26
        cv2.putText(img, name, (x + 2, yy + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(img, (x + BAR_LABEL_W, yy + 2),
                      (x + BAR_LABEL_W + w, yy + 18), (60, 60, 60), -1)
        cv2.rectangle(img, (x + BAR_LABEL_W, yy + 2),
                      (x + BAR_LABEL_W + int(w * p), yy + 18),
                      ACTION_COLORS[i], -1)
        cv2.putText(img, f"{p:.2f}", (x + BAR_LABEL_W + w + 4, yy + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1,
                    cv2.LINE_AA)


def bottom_text(model: str, rec: dict) -> str:
    if rec["success"]:
        return (f"{model.upper()} SUCCESS | steps {rec['steps']} | "
                f"final {rec['final_dist']:.1f} m | min {rec['min_dist']:.1f} m"
                f" | stop {rec['stop_dist']:.1f} m")
    return (f"{model.upper()} FAIL: {rec['failure']} | steps {rec['steps']} | "
            f"final {rec['final_dist']:.1f} m | min {rec['min_dist']:.1f} m")


def conclusion(winner: str, loser: str, wrec: dict, lrec: dict,
               wb: dict, lb: dict) -> str:
    head = (f"{winner} succeeds (stop {wrec['stop_dist']:.1f} m in "
            f"{wrec['steps']} steps) while {loser} fails with "
            f"{lrec['failure']} after {lrec['steps']} steps.")
    tail = ""
    if wb.get("final_est_m") is not None and lb.get("max_est_m") is not None:
        tail = (f" Belief estimate error at end: {winner} "
                f"{wb['final_est_m']:.1f} m vs {loser} max "
                f"{lb['max_est_m']:.1f} m.")
    return head + tail


# ------------------------------------------------------------------ rendering


def _load_replay(model: str, eid: str) -> dict:
    p = REPLAY_DIR / f"{model}_{eid.replace(':', '_')}.json"
    if not p.exists():
        raise SystemExit(f"missing replay {p} -- run replay_corrected.py "
                         f"--model {model} first")
    rep = json.loads(p.read_text(encoding="utf-8"))
    assert rep["argmax_fully_consistent"], \
        f"{model} {eid}: replay argmax inconsistent -- refusing to render"
    return rep


def _context() -> dict:
    ctx: dict = {}
    if METRICS_V4V5.exists():
        m = json.loads(METRICS_V4V5.read_text(encoding="utf-8"))
        ctx["v4_v5"] = {"v4": {"sr": m["v4"]["sr"],
                               "by_split": {k: v["sr"] for k, v
                                            in m["v4"]["by_split"].items()}},
                        "v5": {"sr": m["v5"]["sr"],
                               "by_split": {k: v["sr"] for k, v
                                            in m["v5"]["by_split"].items()}}}
    if METRICS_V3.exists():
        m = json.loads(METRICS_V3.read_text(encoding="utf-8"))
        ctx["v3_corrected_val"] = {"sr": m["v3_corrected_val"]["sr"],
                                   "by_split": {k: v["sr"] for k, v in
                                                m["v3_corrected_val"]
                                                ["by_split"].items()}}
    return ctx


def render_case(case: dict, recs: dict[str, dict],
                replays: dict[str, dict], raster: MapRaster,
                tmp_root: Path, out_dir: Path) -> dict:
    started = time.time()
    eid = case["episode_id"]
    winner, loser = case["winner_model"], case["loser_model"]
    wrec, lrec = recs[winner], recs[loser]
    assert wrec["map"] == lrec["map"]
    name = f"{eid.replace(':', '_')}_{winner}_vs_{loser}"
    difficulty = wrec.get("length_group") or "n/a"

    bounds = {"left": raster.left, "right": raster.right,
              "bottom": raster.bottom, "top": raster.top}
    pts_lists = [[[t["x"], t["y"]] for t in r["trajectory"]]
                 for r in (wrec, lrec)]
    win = window_for(pts_lists, wrec["target"], bounds,
                     MAP_X1 - MAP_X0, MAP_Y1 - MAP_Y0)
    x0, y0, x1, y1 = win
    c0, r0 = raster.pixel(x0, y1)
    c1, r1 = raster.pixel(x1, y0)
    c0, r0 = max(int(c0), 0), max(int(r0), 0)
    c1, r1 = min(int(c1), raster.image.shape[1]), \
        min(int(r1), raster.image.shape[0])
    panel = MapPanel(raster.image[r0:r1, c0:c1].copy(), x0, y0, x1, y1)
    base = panel.canvas()
    target = wrec["target"]
    tx, ty = panel.xy(*target)
    cv2.circle(base, (tx, ty), int(STOP_RADIUS * panel.s), (0, 0, 200), 1,
               cv2.LINE_AA)
    cv2.putText(base, "20 m", (tx + int(STOP_RADIUS * panel.s) + 4, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1, cv2.LINE_AA)
    cv2.drawMarker(base, (tx, ty), (0, 0, 255), cv2.MARKER_STAR, 18, 2)
    sx, sy = panel.xy(*wrec["start"][:2])
    cv2.circle(base, (sx, sy), 6, (60, 200, 60), -1, cv2.LINE_AA)
    cv2.putText(base, "start", (sx + 8, sy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (60, 200, 60), 1, cv2.LINE_AA)
    # final-position markers + legend (model colours)
    lx0, ly0 = 8, 18
    for m, rec in ((winner, wrec), (loser, lrec)):
        last = rec["trajectory"][-1]
        fx, fy = panel.xy(last["x"], last["y"])
        cv2.drawMarker(base, (fx, fy), MODEL_COLORS[m], cv2.MARKER_TILTED_CROSS,
                       14, 1, cv2.LINE_AA)
        cv2.rectangle(base, (lx0, ly0), (lx0 + 14, ly0 + 14),
                      MODEL_COLORS[m], -1)
        cv2.putText(base, f"{m} {'(win)' if m == winner else '(lose)'}",
                    (lx0 + 20, ly0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    MODEL_COLORS[m], 1, cv2.LINE_AA)
        ly0 += 20

    n = max(len(wrec["trajectory"]), len(lrec["trajectory"]))
    frames = []
    for i in range(n):
        frame = base.copy()
        for slot, (m, rec) in (("winner", (winner, wrec)),
                               ("loser", (loser, lrec))):
            traj = rec["trajectory"]
            idx, ended = step_for(i, len(traj))
            st = traj[idx]
            color = MODEL_COLORS[m]
            pts = np.asarray([panel.xy(t["x"], t["y"])
                              for t in traj[:idx + 1]], np.int32)
            if len(pts) >= 2:
                cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)
            cx, cy = panel.xy(st["x"], st["y"])
            hx = int(cx + 22 * np.cos(st["yaw"]))
            hy = int(cy - 22 * np.sin(st["yaw"]))
            cv2.arrowedLine(frame, (cx, cy), (hx, hy), (0, 165, 255), 2,
                            cv2.LINE_AA, tipLength=0.35)
            cv2.circle(frame, (cx, cy), 4, (0, 165, 255), -1, cv2.LINE_AA)
            if st.get("estimate") is not None:
                ex, ey = panel.xy(st["estimate"][0], st["estimate"][1])
                cv2.drawMarker(frame, (ex, ey), color, cv2.MARKER_TILTED_CROSS,
                               10, 1, cv2.LINE_AA)
            # observation panel
            ox, oy_ = OBS_SLOTS[slot]
            obs = raster.heading_crop(float(st["x"]), float(st["y"]),
                                      float(st["yaw"]))
            obs = cv2.resize(obs, (OBS_SIZE, OBS_SIZE),
                             interpolation=cv2.INTER_NEAREST)
            frame[oy_:oy_ + OBS_SIZE, ox:ox + OBS_SIZE] = obs
            cv2.rectangle(frame, (ox, oy_), (ox + OBS_SIZE, oy_ + OBS_SIZE),
                          (90, 90, 90), 1)
            if ended:
                badge = (f"ENDED @ step {idx + 1} — "
                         + ("SUCCESS" if rec["success"]
                            else f"FAIL: {rec['failure']}"))
                bcolor = (90, 220, 90) if rec["success"] else (90, 140, 240)
            else:
                badge = f"{m} UAV view (heading-forward, 100 m)"
                bcolor = (255, 255, 255)
            (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.42, 1)
            cv2.rectangle(frame, (ox + 4, oy_ + 4),
                          (ox + 12 + tw, oy_ + 14 + th), (20, 20, 20), -1)
            cv2.putText(frame, badge, (ox + 8, oy_ + 9 + th),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, bcolor, 1, cv2.LINE_AA)
            # probability bars (real softmax from replay)
            bx, by = BAR_SLOTS[slot]
            rep = replays[m]
            probs_by_step = {s["step"]: s["probs"] for s in rep["trajectory"]}
            draw_prob_bars(frame, probs_by_step[st["step"]], bx, by)
            dist = float(st["true_dist_m"])
            cv2.putText(frame, f"{m} action {CLASSES[st['action']]} | "
                               f"dist {dist:.1f} m", (bx, by + 4 * 26 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.putText(frame, bottom_text(m, rec), (bx, by + 4 * 26 + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1,
                        cv2.LINE_AA)
        # bottom bar
        cv2.rectangle(frame, (0, 650), (W, H), (25, 25, 25), -1)
        title = (f"{eid} | {wrec['split']} | {wrec['map']} | difficulty "
                 f"{difficulty} | target marker: visualization only — not "
                 f"policy input")
        cv2.putText(frame, title, (10, 676), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, bottom_text(winner, wrec), (10, 702),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, MODEL_COLORS[winner], 1,
                    cv2.LINE_AA)
        lt = bottom_text(loser, lrec)
        (tw2, _), _ = cv2.getTextSize(lt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, lt, (W - tw2 - 10, 702), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, MODEL_COLORS[loser], 1, cv2.LINE_AA)
        for _ in range(HOLD):
            frames.append(frame)
    for _ in range(HOLD * TAIL):
        frames.append(frames[-1])

    tmp = Path(tmp_root) / name
    tmp.mkdir(parents=True, exist_ok=True)
    for fi, f in enumerate(frames):
        cv2.imwrite(str(tmp / f"{fi:05d}.png"), f)
    cat_dir = out_dir / case["category"]
    cat_dir.mkdir(parents=True, exist_ok=True)
    mp4 = cat_dir / f"{name}.mp4"
    tmp_mp4 = mp4.with_name(f".{mp4.name}.tmp.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate",
                    str(FPS), "-i", str(tmp / "%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "veryfast",
                    "-movflags", "+faststart", str(tmp_mp4)], check=True)
    os.replace(tmp_mp4, mp4)
    shutil.rmtree(tmp, ignore_errors=True)

    COVERS.mkdir(parents=True, exist_ok=True)
    cover = COVERS / f"{name}.png"
    cv2.imwrite(str(cover), cv2.resize(frames[0], (640, 360),
                                       interpolation=cv2.INTER_AREA))

    # metadata (computed from real numbers only)
    wb = {"final_est_m": None, "est_to_target_m": None}
    if wrec["trajectory"]:
        wlast = wrec["trajectory"][-1]
        wb = {"final_est_m": wlast.get("est_distance_m"),
              "est_to_target_m": (float(((wlast["estimate"][0] - target[0]) ** 2
                                         + (wlast["estimate"][1]
                                            - target[1]) ** 2) ** 0.5)
                                  if wlast.get("estimate") is not None
                                  else None)}
    lb = {"max_est_m": None}
    ests = [s.get("est_distance_m") for s in lrec["trajectory"]
            if s.get("est_distance_m") is not None]
    if ests:
        lb = {"max_est_m": max(ests)}
    meta = {
        "episode_id": eid, "category": case["category"],
        "split": wrec["split"], "map": wrec["map"], "difficulty": difficulty,
        "winner_model": winner, "loser_model": loser,
        "models": {}, "context": _context(),
        "conclusion": conclusion(winner, loser, wrec, lrec, wb, lb),
        "video": {"path": str(mp4), "sha256": sha256(mp4),
                  "bytes": mp4.stat().st_size,
                  "duration_s": round(len(frames) / FPS, 2),
                  "frames": len(frames), "fps": FPS, "hold": HOLD,
                  "tail": TAIL},
        "generation": {
            "scripts": {f"{s.name}": sha256(s) for s in
                        (HERE / "select_cases.py", HERE / "replay_corrected.py",
                         HERE / "make_comparison_videos.py")},
            "git_commit": subprocess.run(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"],
                capture_output=True, text=True).stdout.strip(),
            "params": {"W": W, "H": H, "layout_version": 1,
                       "model_colors": {m: list(c) for m, c
                                        in MODEL_COLORS.items()},
                       "crf": 26, "preset": "veryfast", "pix_fmt": "yuv420p"}},
        "target_marker_note": "visualization only — not policy input",
        "render_seconds": round(time.time() - started, 2),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    for m, rec in (winner, wrec), (loser, lrec):
        rep = replays[m]
        src = Path(rep["source_episode"])
        meta["models"][m] = {
            "ckpt": rep["checkpoint"], "ckpt_sha256": rep["checkpoint_sha256"],
            "success": rec["success"], "failure": rec["failure"],
            "stop_dist": rec.get("stop_dist"), "final_dist": rec["final_dist"],
            "min_dist": rec["min_dist"], "steps": rec["steps"],
            "spl": rec.get("spl"), "osr": rec.get("osr"),
            "trajectory_ref": str(src), "trajectory_sha256": sha256(src),
            "replay_ref": str(REPLAY_DIR / f"{m}_{eid.replace(':', '_')}.json"),
            "replay_sha256": sha256(REPLAY_DIR / f"{m}_{eid.replace(':', '_')}.json"),
            "replay_argmax_consistent": rep["argmax_fully_consistent"]}
    META_DIR.mkdir(parents=True, exist_ok=True)
    (META_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2,
                                                      ensure_ascii=False)
                                           + "\n", encoding="utf-8")
    return meta


def _freeze_generation_artifacts(selection: Path) -> None:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    for s in ("select_cases.py", "replay_corrected.py",
              "make_comparison_videos.py"):
        shutil.copy2(HERE / s, GEN_DIR / s)
    if selection.exists():
        shutil.copy2(selection, GEN_DIR / "selection.json")
    params = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "selection_source": str(selection)}
    (GEN_DIR / "render_params.json").write_text(
        json.dumps(params, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", default=str(SELECTION))
    parser.add_argument("--category", default="")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("CITYNAV_DATA"):
        raise SystemExit("CITYNAV_DATA is not set")
    selection = Path(args.selection)
    if not selection.exists():
        raise SystemExit(f"selection.json missing: {selection}")
    data = json.loads(selection.read_text(encoding="utf-8"))
    cases: list[dict] = []
    for cat, cdata in data.get("categories", {}).items():
        if args.category and cat != args.category:
            continue
        for p in cdata.get("selected", []):
            if args.only and p["episode_id"] not in set(args.only):
                continue
            cases.append(p)
    if not cases:
        raise SystemExit("no cases to render")
    _freeze_generation_artifacts(selection)
    rasters: dict[str, MapRaster] = {}
    tmp_root = Path(tempfile.mkdtemp(prefix="comparison_video_"))
    for case in cases:
        eid = case["episode_id"]
        winner, loser = case["winner_model"], case["loser_model"]
        name = f"{eid.replace(':', '_')}_{winner}_vs_{loser}"
        mp4 = CASE_ROOT / case["category"] / f"{name}.mp4"
        meta_path = META_DIR / f"{name}.json"
        if not args.force and mp4.exists() and meta_path.exists():
            print(f"skip {eid} (rendered)", flush=True)
            continue
        reps = {winner: _load_replay(winner, eid),
                loser: _load_replay(loser, eid)}
        recs = {winner: json.loads(Path(reps[winner]["source_episode"])
                                   .read_text(encoding="utf-8")),
                loser: json.loads(Path(reps[loser]["source_episode"])
                                  .read_text(encoding="utf-8"))}
        assert recs[winner]["map"] == recs[loser]["map"], \
            f"{eid}: map mismatch between {winner} and {loser}"
        raster = rasters.setdefault(recs[winner]["map"],
                                    MapRaster.load(recs[winner]["map"]))
        meta = render_case(case, recs, reps, raster, tmp_root, CASE_ROOT)
        print(f"rendered {eid} {winner} vs {loser} "
              f"{meta['video']['duration_s']}s "
              f"{meta['video']['bytes'] / 1e6:.2f} MB | "
              f"winner={meta['models'][winner]['success']} "
              f"loser={meta['models'][loser]['failure']}", flush=True)
    shutil.rmtree(tmp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
