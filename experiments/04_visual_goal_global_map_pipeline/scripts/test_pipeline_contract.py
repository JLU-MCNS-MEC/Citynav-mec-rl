"""Contract tests for the visual-goal / global-map pipeline."""

from __future__ import annotations

import inspect
import math

import numpy as np

from pipeline import (
    HISTORY_LEN,
    CandidateLibrary,
    MapRaster,
    belief_features,
    build_candidates,
    history_window,
    view_area_corners,
    wrap,
)


def synthetic_raster() -> MapRaster:
    img = np.zeros((1000, 1000, 3), np.uint8)
    img[::10, :, :] = 128  # simple texture
    img[:, ::10, :] = 200
    return MapRaster("toy", img, 0.0, 0.0, 100.0, 100.0, 0.1, np.ones((100, 100), bool))


def test_view_corners_are_square_and_centered():
    corners = np.asarray(view_area_corners(50.0, 50.0, 0.7, 50.0))
    sides = np.linalg.norm(np.diff(np.vstack([corners, corners[:1]]), axis=0), axis=1)
    assert np.allclose(sides, 100.0)
    assert np.allclose(corners.mean(axis=0), [50.0, 50.0])


def test_heading_crop_shape():
    raster = synthetic_raster()
    crop = raster.heading_crop(50.0, 50.0, 0.0, half=50.0, out=224)
    assert crop.shape == (224, 224, 3)
    assert crop.dtype == np.uint8


def test_north_crop_shape_and_padding():
    raster = synthetic_raster()
    crop = raster.north_crop(99.0, 99.0, half=20.0, out=64)
    assert crop.shape == (64, 64, 3)


def test_belief_has_no_target_coordinate_argument():
    params = inspect.signature(belief_features).parameters
    assert "target_xy" not in params and "target" not in params
    assert list(params)[:4] == ["library", "target_emb", "pose_xy", "yaw"]


def test_belief_dimensions_and_topk():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(20, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    lib = CandidateLibrary("toy", rng.uniform(0, 100, size=(20, 2)).astype(np.float32), emb)
    ref = emb[3].copy()
    feature, info = belief_features(lib, ref, np.array([50.0, 50.0]), 0.0, topk=5)
    assert feature.shape == (12,)
    assert len(info["top_indices"]) == 5
    assert info["top_indices"][0] == 3


def test_history_is_causal_and_masked():
    rng = np.random.default_rng(1)
    feats = rng.normal(size=(30, 16)).astype(np.float32)
    states = np.stack([np.arange(30, dtype=float), np.zeros(30), np.zeros(30)], axis=1)
    actions = np.zeros(30, dtype=np.int64)
    hist = history_window(feats, states, actions, 20)
    assert hist["history_vis"].shape == (HISTORY_LEN, 16)
    assert int(hist["history_mask"].sum()) == HISTORY_LEN
    # future mutations must not change the history
    feats2, states2, actions2 = feats.copy(), states.copy(), actions.copy()
    feats2[21:] = 0
    states2[21:] = 999
    actions2[21:] = 3
    hist2 = history_window(feats2, states2, actions2, 20)
    for key in hist:
        assert np.array_equal(hist[key], hist2[key])
    short = history_window(feats, states, actions, 3)
    assert int(short["history_mask"].sum()) == 3


def test_history_relative_pose_is_in_current_frame():
    feats = np.zeros((5, 4), np.float32)
    states = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [5, 0, 0], [0, 0, 0]], float)
    actions = np.zeros(5, np.int64)
    hist = history_window(feats, states, actions, 4)
    # the past state is 5 m ahead of the current one in world x; with yaw=0 the
    # relative offset in the current heading frame is +5 m forward
    assert np.allclose(hist["history_pose"][0], [5.0, 0.0, 0.0], atol=1e-6)


def test_valid_fraction_counts_out_of_bounds_as_invalid():
    img = np.zeros((1000, 1000, 3), np.uint8)
    validity = np.ones((100, 100), bool)
    validity[:, 50:] = False                       # x >= 50 m is nodata
    raster = MapRaster("toy", img, 0.0, 0.0, 100.0, 100.0, 0.1, validity)
    assert abs(raster.valid_fraction(25.0, 50.0, 20.0) - 1.0) < 1e-6
    assert abs(raster.valid_fraction(45.0, 50.0, 20.0) - 0.625) < 1e-6
    assert raster.valid_fraction(95.0, 50.0, 20.0) < 0.5   # half outside raster
    assert raster.height_valid(25.0, 50.0)
    assert not raster.height_valid(75.0, 50.0)


class _StubEncoder:
    def embed(self, images):
        out = np.zeros((len(images), 8), np.float32)
        out[:, 0] = 1.0
        return out


def test_build_candidates_skips_mostly_invalid_patches():
    img = np.zeros((1000, 1000, 3), np.uint8)
    validity = np.ones((100, 100), bool)
    validity[:, 50:] = False
    raster = MapRaster("toy", img, 0.0, 0.0, 100.0, 100.0, 0.1, validity)
    lib = build_candidates(raster, _StubEncoder(), stride=20.0, half=20.0, min_valid=0.6)
    assert len(lib.positions) > 0
    assert lib.valid_fraction is not None
    assert np.all(lib.valid_fraction >= 0.6)
    assert np.all(lib.positions[:, 0] < 50.0)
    assert np.allclose(np.linalg.norm(lib.embeddings, axis=1), 1.0)


def test_belief_topk_does_not_change_feature_but_extends_ranking():
    rng = np.random.default_rng(2)
    emb = rng.normal(size=(40, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    positions = np.stack([np.arange(40) * 5.0, np.zeros(40)], axis=1).astype(np.float32)
    lib = CandidateLibrary("toy", positions, emb)
    ref = emb[7].copy()
    f5, i5 = belief_features(lib, ref, np.array([0.0, 0.0]), 0.0, topk=5)
    f10, i10 = belief_features(lib, ref, np.array([0.0, 0.0]), 0.0, topk=10)
    assert np.array_equal(f5, f10)                  # the 12-d belief is top-5 only
    assert len(i5["top_indices"]) == 5
    assert len(i10["top_indices"]) == 10
    assert len(i10["rank_distances"]) == 40
    assert i10["top_indices"][0] == 7
    # Recall@K must come from the full ranking, not from the 5-element top list
    k10 = 10
    assert len(i10["rank_distances"][:k10]) == k10


def test_wrap_range():
    for a in (-10.0, -math.pi, 0.0, 7.0):
        assert -math.pi <= wrap(a) < math.pi
