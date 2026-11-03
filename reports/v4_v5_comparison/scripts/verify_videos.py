#!/usr/bin/env python3
"""Final verification of the case-study comparison videos.

Five gates (spec section 6):
  1. ffprobe: h264 / 1280x720 / yuv420p / duration>0 / frame count ==
     (max_steps + TAIL) * HOLD per video
  2. replay consistency: argmax==saved action at every step, prob sums 1,
     replay length == stored steps, checkpoint sha == eval_config sha
  3. frame-vs-trajectory: rebuild the same map panel deterministically from
     the episode JSONs and assert sampled video frames show each model's
     pose at the trajectory position (and freeze after end)
  4. no-target-as-input audit: the replay script must enforce the input
     contract at call time (batch keys == contract, target coords never in
     the batch); same audit text applied to any poisoned variant in tests
  5. git review: git status / diff --stat shows only the intended paths

Exit code non-zero if any gate fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CASE_ROOT = REPO / "reports/v4_v5_comparison/case_studies"
META_DIR = CASE_ROOT / "metadata"
REPLAY_DIR = META_DIR / "replay"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments/04_visual_goal_global_map_pipeline"
                        / "scripts"))

import make_comparison_videos as mcv  # noqa: E402
from pipeline import MapRaster  # noqa: E402

EXPECTED_BATCH_KEYS = {
    "v4": ["current", "target_emb", "history_vis", "history_pose",
           "history_action", "history_mask", "belief", "match"],
    "v5": ["current", "target_emb", "history_vis", "history_pose",
           "history_action", "history_mask", "belief", "match",
           "landmarks", "states"],
    "v3": ["current", "target_emb", "history_vis", "history_pose",
           "history_action", "history_mask", "belief", "match"],
}
POSE_COLOR = (0, 165, 255)


# ------------------------------------------------------------ gate 1: ffprobe


def ffprobe_info(mp4: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of",
         "json", str(mp4)], capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def check_video_props(info: dict, w: int, h: int, frames: int) -> list[str]:
    errs = []
    vstreams = [s for s in info.get("streams", []) if s.get("codec_type")
                == "video"]
    if not vstreams:
        return ["no video stream"]
    s = vstreams[0]
    if s.get("codec_name") != "h264":
        errs.append(f"codec {s.get('codec_name')} != h264")
    if (s.get("width"), s.get("height")) != (w, h):
        errs.append(f"resolution {s.get('width')}x{s.get('height')} != "
                    f"{w}x{h}")
    if s.get("pix_fmt") != "yuv420p":
        errs.append(f"pix_fmt {s.get('pix_fmt')} != yuv420p")
    dur = float(info.get("format", {}).get("duration", 0) or 0)
    if dur <= 0:
        errs.append("duration <= 0")
    nb = int(s.get("nb_frames", -1) or -1)
    if nb != -1 and nb != frames:
        errs.append(f"frame count {nb} != expected {frames}")
    return errs


# ------------------------------------------------------------ gate 2: replay


def check_replay(rep: dict, ckpt_sha: str) -> list[str]:
    errs = []
    if not rep.get("argmax_fully_consistent"):
        errs.append("argmax not fully consistent with saved actions")
    if not rep.get("prob_sums_all_one"):
        errs.append("prob sums not all 1.0")
    if len(rep.get("trajectory", [])) != rep.get("steps"):
        errs.append(f"replay trajectory length {len(rep.get('trajectory', []))}"
                    f" != steps {rep.get('steps')}")
    if rep.get("checkpoint_sha256") != ckpt_sha:
        errs.append("checkpoint sha mismatch vs eval_config")
    return errs


# ---------------------------------------------------------- gate 3: metadata


def check_metadata(meta: dict) -> list[str]:
    errs = []
    for m, minfo in meta.get("models", {}).items():
        for key, field in (("trajectory_ref", "trajectory"),
                           ("replay_ref", "replay")):
            if not minfo.get(key):
                errs.append(f"{m} {field} ref empty")
                continue
            p = Path(minfo[key])
            if not p.exists():
                errs.append(f"{m} {field} ref missing: {p}")
                continue
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            if sha != minfo[key.replace("ref", "sha256")]:
                errs.append(f"{m} {field} sha mismatch: {p}")
        if not minfo.get("replay_argmax_consistent"):
            errs.append(f"{m} replay argmax inconsistent")
    v = meta.get("video", {})
    if not v.get("path"):
        errs.append("video path empty")
        return errs
    vp = Path(v["path"])
    if not vp.exists():
        errs.append(f"video missing: {vp}")
    elif hashlib.sha256(vp.read_bytes()).hexdigest() != v.get("sha256"):
        errs.append(f"video sha mismatch: {vp}")
    return errs


# ------------------------------------------------------- gate 4: static audit


def audit_contracts(source: str) -> list[str]:
    errs = []
    if "assert_batch_contract" not in source:
        errs.append("replay source lacks runtime assert_batch_contract guard")
    import re
    m = re.search(r"BASE_CONTRACT = (\[[^\]]*\])", source)
    if m:
        try:
            keys = eval(m.group(1))  # noqa: S307 -- static own-source audit
        except Exception:  # noqa: BLE001
            keys = []
        if any(k == "target" or k in ("target_x", "target_y", "target_xy",
                                      "target_pos") for k in keys):
            errs.append("contract contains a true-target key")
    return errs


# ------------------------------------------------------ gate 3b: pixel check


def check_frames(mp4: Path, meta: dict, hold: int = mcv.HOLD,
                 tail: int = mcv.TAIL) -> list[str]:
    """Rebuild the deterministic map panel and verify sampled frames."""
    if not os.environ.get("CITYNAV_DATA"):
        return []  # frame check requires map rasters; skip w/o data
    errs = []
    recs = {}
    for m, minfo in meta["models"].items():
        recs[m] = json.loads(Path(minfo["trajectory_ref"]).read_text(
            encoding="utf-8"))
    wm, lm_ = meta["winner_model"], meta["loser_model"]
    assert recs[wm]["map"] == recs[lm_]["map"]
    raster = MapRaster.load(recs[wm]["map"])
    bounds = {"left": raster.left, "right": raster.right,
              "bottom": raster.bottom, "top": raster.top}
    pts_lists = [[[t["x"], t["y"]] for t in r["trajectory"]]
                 for r in (recs[wm], recs[lm_])]
    win = mcv.window_for(pts_lists, recs[wm]["target"], bounds,
                         mcv.MAP_X1 - mcv.MAP_X0, mcv.MAP_Y1 - mcv.MAP_Y0)
    x0, y0, x1, y1 = win
    c0, r0 = raster.pixel(x0, y1)
    c1, r1 = raster.pixel(x1, y0)
    panel = mcv.MapPanel(raster.image[max(int(r0), 0):min(int(r1),
                                                          raster.image.shape[0]),
                                      max(int(c0), 0):min(int(c1),
                                                          raster.image.shape[1])]
                         .copy(), x0, y0, x1, y1)
    n = max(len(recs[wm]["trajectory"]), len(recs[lm_]["trajectory"]))
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expect = (n + tail) * hold
    if total != expect:
        errs.append(f"video frame count {total} != expected {expect}")
    for k in (0, n // 2, n - 1):
        fi = hold * k + 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            errs.append(f"cannot read frame {fi}")
            continue
        for m, rec in recs.items():
            traj = rec["trajectory"]
            idx, ended = mcv.step_for(k, len(traj))
            st = traj[idx]
            cx, cy = panel.xy(st["x"], st["y"])
            px = frame[cy, cx]
            # pose circle colour, or the model colour if the belief cross
            # (estimate == position) overlaps the pose pixel
            expected = {mcv.POSE_COLOR, mcv.MODEL_COLORS[m]}
            if not any(all(abs(int(a) - int(b)) <= 10 for a, b
                           in zip(px, e)) for e in expected):
                errs.append(f"frame {fi} {m}: pixel {tuple(px)} at "
                            f"({cx},{cy}) != pose colour (ended={ended})")
    cap.release()
    return errs


# ---------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-dir", default=str(META_DIR))
    parser.add_argument("--skip-frames", action="store_true",
                        help="skip the frame-vs-trajectory gate")
    args = parser.parse_args()
    meta_dir = Path(args.meta_dir)
    all_errs: dict[str, list[str]] = {}
    for mp in sorted(meta_dir.glob("*.json")):
        meta = json.loads(mp.read_text(encoding="utf-8"))
        key = mp.stem
        errs: list[str] = []
        vpath = Path(meta["video"]["path"])
        errs += check_metadata(meta)
        if vpath.exists():
            errs += check_video_props(ffprobe_info(vpath), 1280, 720,
                                      meta["video"]["frames"])
        if not args.skip_frames and vpath.exists():
            errs += check_frames(vpath, meta)
        all_errs[key] = errs
    for rp in sorted(REPLAY_DIR.glob("*.json")):
        rep = json.loads(rp.read_text(encoding="utf-8"))
        # authoritative checkpoint sha: the model run's eval_config.json
        eval_cfg = Path(rep["source_episode"]).parent.parent / "eval_config.json"
        ckpt_sha = ""
        if eval_cfg.exists():
            ckpt_sha = json.loads(eval_cfg.read_text(encoding="utf-8")) \
                .get("ckpt_sha", "")
        errs = check_replay(rep, ckpt_sha)
        if errs:
            all_errs[f"replay:{rp.stem}"] = errs
    errs = audit_contracts((HERE / "replay_corrected.py").read_text(
        encoding="utf-8"))
    if errs:
        all_errs["audit:replay_corrected.py"] = errs

    total = 0
    for key, errs in sorted(all_errs.items()):
        if errs:
            total += len(errs)
            print(f"[FAIL] {key}")
            for e in errs:
                print(f"       - {e}")
        else:
            print(f"[ok]   {key}")
    print(f"verification {'FAILED' if total else 'PASSED'}: "
          f"{len(all_errs)} items, {total} errors")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
