#!/usr/bin/env python3
"""Coordinate transforms for the online goal matching (Stage 09).

Two frames are used:

* heading frame (``MapRaster.heading_crop``): forward is up, left is left;
  the crop covers 100 m x 100 m in 224 px;
* north-up frame (``MapRaster.north_crop`` / the rotated observation): +y is up.

The mapping below is derived from ``pipeline.view_area_corners``:
corner order (front-left, front-right, back-right, back-left) is mapped to
(top-left, top-right, bottom-right, bottom-left), i.e. a pure heading-up view
without mirroring.  ``self_test`` checks all four cardinal headings against the
perspective transform actually used by ``heading_crop``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "04_visual_goal_global_map_pipeline/scripts"))
from pipeline import view_area_corners  # noqa: E402

CUR_HALF_M = 50.0
GOAL_HALF_M = 48.0
OUT = 224


def scale_for(half: float, out: int = OUT) -> float:
    return 2.0 * half / (out - 1)


CUR_SCALE = scale_for(CUR_HALF_M)      # 0.4484 m/px
GOAL_SCALE = scale_for(GOAL_HALF_M)    # 0.4305 m/px


def center_px(out: int = OUT) -> float:
    return (out - 1) / 2.0


def heading_pixel(x: float, y: float, yaw: float, xw: float, yw: float,
                  half: float = CUR_HALF_M, out: int = OUT) -> tuple[float, float]:
    """World point -> (col, row) in the heading-up crop."""
    dx, dy = xw - x, yw - y
    f = dx * math.cos(yaw) + dy * math.sin(yaw)
    l = -dx * math.sin(yaw) + dy * math.cos(yaw)
    col = (half - l) / (2 * half) * (out - 1)
    row = (half - f) / (2 * half) * (out - 1)
    return col, row


def heading_world(x: float, y: float, yaw: float, col: float, row: float,
                  half: float = CUR_HALF_M, out: int = OUT) -> tuple[float, float]:
    f = half - row / (out - 1) * 2 * half
    l = half - col / (out - 1) * 2 * half
    return x + f * math.cos(yaw) - l * math.sin(yaw), y + f * math.sin(yaw) + l * math.cos(yaw)


def north_pixel(x: float, y: float, xw: float, yw: float,
                scale: float = CUR_SCALE, out: int = OUT) -> tuple[float, float]:
    c = center_px(out)
    return c + (xw - x) / scale, c - (yw - y) / scale


def north_world(x: float, y: float, col: float, row: float,
                scale: float = CUR_SCALE, out: int = OUT) -> tuple[float, float]:
    c = center_px(out)
    return x + (col - c) * scale, y - (row - c) * scale


def rotation_matrix(yaw: float, out: int = OUT) -> np.ndarray:
    """Matrix that moves a heading-frame pixel to the north-up frame.

    Heading-up is the north-up image rotated counter-clockwise by ``yaw``
    (cv2 positive angle); to undo it we rotate by ``-yaw``.
    """
    c = center_px(out)
    # heading-up is the north-up frame rotated by (yaw - 90 deg) in cv2 terms
    return cv2.getRotationMatrix2D((c, c), math.degrees(yaw) - 90.0, 1.0)


def to_northup(img: np.ndarray, yaw: float, out: int = OUT) -> np.ndarray:
    return cv2.warpAffine(img, rotation_matrix(yaw, out), (out, out),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))


def goal_to_current_scale(goal: np.ndarray, out: int = OUT) -> np.ndarray:
    """96 m goal image -> the 100 m observation scale, centre-padded."""
    span = int(round(out * GOAL_HALF_M / CUR_HALF_M))     # 214
    small = cv2.resize(goal, (span, span), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out, out, 3), np.uint8)
    off = (out - span) // 2
    canvas[off:off + span, off:off + span] = small
    return canvas


def self_test(verbose: bool = True) -> dict:
    """Verify heading_pixel and the north-up rotation on four headings."""
    from pipeline import MapRaster

    raster = MapRaster.load("birmingham_block_1")
    x, y = 212.0, 671.0
    out = {}
    for deg in (0, 90, 180, 270):
        yaw = math.radians(deg)
        # a point 30 m forward and 10 m left of the UAV
        f, l = 30.0, 10.0
        xw = x + f * math.cos(yaw) - l * math.sin(yaw)
        yw = y + f * math.sin(yaw) + l * math.cos(yaw)
        col, row = heading_pixel(x, y, yaw, xw, yw)
        # perspective transform actually used by heading_crop
        corners = view_area_corners(x, y, yaw, CUR_HALF_M)
        src = np.stack(raster.pixel([p[0] for p in corners], [p[1] for p in corners]), axis=1)
        dst = np.array([[0, 0], [OUT - 1, 0], [OUT - 1, OUT - 1], [0, OUT - 1]], np.float32)
        m = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
        cpx, rpx = raster.pixel(xw, yw)
        pts = np.array([[float(np.ravel(cpx)[0]), float(np.ravel(rpx)[0])]], np.float32).reshape(-1, 1, 2)
        pt = cv2.perspectiveTransform(pts, m).reshape(-1, 2)[0]
        # rotated heading pixel -> north-up pixel, compared to the direct formula
        mrot = rotation_matrix(yaw)
        rot = cv2.transform(np.array([[col, row]], np.float32).reshape(-1, 1, 2), mrot).reshape(-1, 2)[0]
        ncol, nrow = north_pixel(x, y, xw, yw)
        out[deg] = {"heading": (col, row), "perspective": (float(pt[0]), float(pt[1])),
                    "rotated": (float(rot[0]), float(rot[1])), "north": (ncol, nrow)}
        if verbose:
            print(f"yaw {deg:3d}: heading={col:7.2f},{row:7.2f} | persp={pt[0]:7.2f},{pt[1]:7.2f}"
                  f" | rotated={rot[0]:7.2f},{rot[1]:7.2f} | north={ncol:7.2f},{nrow:7.2f}")
    for deg, v in out.items():
        assert max(abs(v["heading"][0] - v["perspective"][0]),
                   abs(v["heading"][1] - v["perspective"][1])) < 0.05, f"heading mismatch {deg}"
        assert max(abs(v["rotated"][0] - v["north"][0]),
                   abs(v["rotated"][1] - v["north"][1])) < 1.5, f"rotation mismatch {deg}"
    if verbose:
        print("self_test: heading/perspective and rotation/north agree on 0/90/180/270")
    return out


if __name__ == "__main__":
    self_test()
