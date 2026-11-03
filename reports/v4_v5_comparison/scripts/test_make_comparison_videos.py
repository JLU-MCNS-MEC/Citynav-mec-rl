"""Tests for make_comparison_videos.py pure helpers (no CITYNAV_DATA/GPU)."""
from __future__ import annotations

import numpy as np

from make_comparison_videos import (MapPanel, bottom_text, conclusion,
                                    draw_prob_bars, step_for, window_for)

BOUNDS = {"left": 0.0, "right": 1000.0, "bottom": 0.0, "top": 1000.0}


def test_window_for_centers_and_clamps():
    pts = [[100.0, 100.0], [160.0, 140.0]]
    x0, y0, x1, y1 = window_for([pts], [130.0, 120.0], BOUNDS, 640, 640)
    # min span 140, aspect 1 (square panel), pad 35
    assert x1 - x0 >= 140.0 and y1 - y0 >= 140.0
    assert x0 <= 100.0 <= x1 and y0 <= 100.0 <= y1
    assert x0 <= 160.0 <= x1 and y0 <= 140.0 <= y1
    assert abs((x1 - x0) - (y1 - y0)) < 1e-9          # square aspect


def test_window_for_clamps_to_raster_bounds():
    pts = [[970.0, 500.0], [995.0, 520.0]]
    x0, y0, x1, y1 = window_for([pts], [980.0, 510.0], BOUNDS, 640, 640)
    assert x1 <= 1000.0 + 1e-9 and y1 <= 1000.0 + 1e-9
    assert x0 >= 0.0 - 1e-9 and y0 >= 0.0 - 1e-9
    assert abs((x1 - x0) - 140.0) < 1e-9            # span preserved


def test_map_panel_xy_roundtrip():
    crop = np.zeros((50, 50, 3), np.uint8)
    panel = MapPanel(crop, 100.0, 200.0, 300.0, 400.0, panel_w=640, panel_h=640)
    # scale: 640 / 200 = 3.2 px per m; world (150,300) is 50 m in from each
    # edge -> (160, 320) px
    px, py = panel.xy(150.0, 300.0)
    assert abs(px - 160.0) < 2 and abs(py - 320.0) < 2
    # top-left world corner (x0, y1) maps near the panel origin
    px0, py0 = panel.xy(100.0, 400.0)
    assert px0 < 10 and py0 < 10
    assert panel.canvas().shape == (720, 1280, 3)


def test_step_for_freezes_after_end():
    idx, ended = step_for(0, 5)
    assert (idx, ended) == (0, False)
    idx, ended = step_for(4, 5)
    assert (idx, ended) == (4, False)
    idx, ended = step_for(5, 5)
    assert (idx, ended) == (4, True)
    idx, ended = step_for(17, 5)
    assert (idx, ended) == (4, True)


def test_draw_prob_bars_draws_in_region():
    img = np.zeros((720, 1280, 3), np.uint8)
    draw_prob_bars(img, [0.9, 0.05, 0.03, 0.02], 648, 300, 240)
    region = img[300:300 + 4 * 26, 648:648 + 95 + 240 + 40, :]
    assert region.sum() > 0                             # labels + bars drawn
    # nothing drawn outside the bar block (left of x=648 stays black)
    assert img[300:404, 0:640].sum() == 0


def test_bottom_text_success_and_failure():
    ok = {"success": True, "failure": "success", "stop_dist": 12.1,
          "steps": 31, "final_dist": 12.1, "min_dist": 5.2, "stopped": True}
    t = bottom_text("V5", ok)
    assert "V5 SUCCESS" in t and "stop 12.1" in t and "steps 31" in t
    bad = {"success": False, "failure": "out_of_bounds", "stop_dist": None,
           "steps": 12, "final_dist": 27.1, "min_dist": 5.2, "stopped": False}
    t = bottom_text("V4", bad)
    assert "V4 FAIL: out_of_bounds" in t and "final 27.1" in t


def test_conclusion_is_computed_from_numbers():
    w = {"success": True, "stop_dist": 12.1, "steps": 31}
    l = {"success": False, "failure": "out_of_bounds", "steps": 12}
    wb = {"final_est_m": 6.1, "est_to_target_m": 7.2}
    lb = {"max_est_m": 38.2}
    c = conclusion("v5", "v4", w, l, wb, lb)
    assert "12.1" in c and "31" in c and "out_of_bounds" in c
    assert "6.1" in c and "38.2" in c and "v5" in c and "v4" in c
