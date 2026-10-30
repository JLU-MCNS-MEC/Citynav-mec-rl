#!/usr/bin/env python3
"""Paired trajectory cases: Transformer-M vs Top-1 geometry on the real map.

4 Transformer-fail / geometry-success, 4 Transformer-success / geometry-fail,
2 both success, 2 both fail (fewer only if a category runs out).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"
STAGE06 = ROOT / "06_closed_loop_success_rate/results/seed42"
sys.path.insert(0, str(ROOT / "04_visual_goal_global_map_pipeline/scripts"))
from pipeline import MapRaster  # noqa: E402


def load_ep(base: Path, eid: str) -> dict:
    return json.loads((base / f"{eid.replace(':', '_')}.json").read_text(encoding="utf-8"))


def main() -> int:
    comp = json.loads((RES / "comparison.json").read_text(encoding="utf-8"))
    manifest = {t["episode_id"]: t for t in
                json.loads((STAGE06 / "manifest.json").read_text(encoding="utf-8"))["tasks"]}
    trows = {json.loads(l)["episode_id"]: json.loads(l)
             for l in (STAGE06 / "journal.jsonl").read_text(encoding="utf-8").splitlines() if l}
    grows = {json.loads(l)["episode_id"]: json.loads(l)
             for l in (RES / "top1/journal.jsonl").read_text(encoding="utf-8").splitlines() if l}
    sets = comp["case_sets"]

    def pick(ids, n, prefer):
        chosen, used_maps = [], set()
        pool = sorted(ids, key=lambda i: (prefer(manifest[i]), i))
        for i in pool:
            m = manifest[i]["map"]
            if m in used_maps:
                continue
            used_maps.add(m)
            chosen.append(i)
            if len(chosen) >= n:
                break
        return chosen

    cases = [("T fail / geom success",
              pick(sets["t_fail_g1_ok"], 4, lambda t: -int(t["teacher_steps"] or 0))),
             ("T success / geom fail",
              pick(sets["t_ok_g1_fail"], 4, lambda t: int(t["teacher_steps"] or 0))),
             ("both success",
              pick([i for i in trows if trows[i]["success"] and grows[i]["success"]],
                   2, lambda t: -int(t["teacher_steps"] or 0))),
             ("both fail",
              pick([i for i in trows if not trows[i]["success"]
                    and not grows[i]["success"]], 2, lambda t: -int(t["teacher_steps"] or 0)))]

    rasters: dict[str, MapRaster] = {}
    fig, axes = plt.subplots(4, 4, figsize=(19, 16))
    for row, (title, ids) in enumerate(cases):
        for col in range(4):
            ax = axes[row][col]
            ax.axis("off")
            if col >= len(ids):
                continue
            eid = ids[col]
            t_rec = load_ep(STAGE06 / "episodes", eid)
            g_rec = load_ep(RES / "top1/episodes", eid)
            name = t_rec["map"]
            if name not in rasters:
                rasters[name] = MapRaster.load(name)
            raster = rasters[name]
            tpts = np.asarray([[s["x"], s["y"]] for s in t_rec["trajectory"]])
            gpts = np.asarray([[s["x"], s["y"]] for s in g_rec["trajectory"]])
            target = np.asarray(t_rec["target"])
            allp = np.vstack([tpts, gpts, target[None]])
            lo = allp.min(axis=0) - 25
            hi = allp.max(axis=0) + 25
            c0, r0 = raster.pixel(lo[0], hi[1])
            c1, r1 = raster.pixel(hi[0], lo[1])
            c0, r0 = max(int(c0), 0), max(int(r0), 0)
            c1, r1 = min(int(c1), raster.image.shape[1]), min(int(r1), raster.image.shape[0])
            crop = raster.image[r0:r1, c0:c1]
            left, right = raster.left + c0 * raster.res, raster.left + c1 * raster.res
            top, bottom = raster.top - r0 * raster.res, raster.top - r1 * raster.res
            ax.imshow(crop, extent=[left, right, bottom, top], origin="upper")
            ax.plot(tpts[:, 0], tpts[:, 1], "-", color="#b23b3b", lw=1.3, label="Transformer-M")
            ax.plot(gpts[:, 0], gpts[:, 1], "-", color="#2f5d8a", lw=1.3, label="Top-1 geom")
            ax.plot(tpts[0, 0], tpts[0, 1], "o", color="#2e8b57", ms=6)
            ax.plot(target[0], target[1], "*", color="#b23b3b", ms=12)
            ax.add_patch(plt.Circle(target, 20, fill=False, color="#b23b3b", lw=1, ls="--"))
            ax.plot(tpts[-1, 0], tpts[-1, 1], "x", color="#b23b3b", ms=9, mew=2)
            ax.plot(gpts[-1, 0], gpts[-1, 1], "x", color="#2f5d8a", ms=9, mew=2)
            if g_rec["trajectory"]:
                s = g_rec["trajectory"][-1]
                ex = s["x"] + s["est_distance_m"] * math.cos(
                    s["yaw"] + math.radians(s["est_bearing_deg"]))
                ey = s["y"] + s["est_distance_m"] * math.sin(
                    s["yaw"] + math.radians(s["est_bearing_deg"]))
                ax.plot(ex, ey, "P", color="#d9822b", ms=8)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=5)
            ax.set_title(f"[{title}] {eid}\nT:{trows[eid]['failure']} "
                         f"({trows[eid]['steps']} steps) | G:{grows[eid]['failure']} "
                         f"({grows[eid]['steps']} steps)", fontsize=7.5)
            ax.axis("on")
            ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([], [], color="#b23b3b", label="Transformer-M trajectory"),
               plt.Line2D([], [], color="#2f5d8a", label="Top-1 geometry trajectory"),
               plt.Line2D([], [], marker="*", color="#b23b3b", ls="", label="true target"),
               plt.Line2D([], [], marker="o", color="#2e8b57", ls="", label="start"),
               plt.Line2D([], [], marker="P", color="#d9822b", ls="", label="estimated goal at STOP")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle("Paired closed-loop trajectories on the real maps (red: Transformer-M, blue: Top-1 geometry)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=2.4, w_pad=1.2)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "trajectory_cases.png", dpi=150)
    plt.close(fig)
    print(f"wrote {FIG / 'trajectory_cases.png'}")
    for title, ids in cases:
        print(title, ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
