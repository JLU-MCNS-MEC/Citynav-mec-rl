"""Tests for replay_corrected.py pure logic (no GPU needed)."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "experiments"
                       / "05_transformer_m_training/scripts"))

import replay_corrected as rc  # noqa: E402


def test_model_specs_have_existing_checkpoints():
    for model in ("v4", "v5", "v3"):
        spec = rc.MODEL_SPECS[model]
        assert spec["ckpt"].exists(), f"{model} ckpt missing"
        assert spec["contract"] == rc.BASE_CONTRACT or \
            spec["contract"] == rc.BASE_CONTRACT + ["landmarks", "states"]


def test_empty_state_matches_run_closed_loop_branch():
    """Mirror run_closed_loop.py empty-candidate block exactly."""
    belief_feat, match_feat, est, dist_est, score, accept = rc.empty_state(
        np.asarray([10.0, 20.0]))
    assert list(belief_feat) == [0, 0, 0, 0, 0, 50.0, 0, 99.0, 0, 0]
    assert list(match_feat) == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert list(est) == [10.0, 20.0]
    assert dist_est == 0.0 and score == 0.0 and accept is False


def test_build_v5_extras_mirrors_landmark_block():
    """Mirror run_closed_loop.py lines 310-319 (v5 landmarks + states)."""
    belief = SimpleNamespace(
        means=np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],
                          [9.0, 10.0], [11.0, 12.0], [13.0, 14.0],
                          [15.0, 16.0]], np.float32),
        weights=np.asarray([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01],
                           np.float32))
    lm, states = rc.build_v5_extras(belief, empty_cands=False,
                                    pos=np.asarray([4.0, 5.0]), yaw=0.7)
    assert lm.shape == (1, 8, 3)
    assert np.allclose(lm[0, :, :2], belief.means)
    assert np.allclose(lm[0, :, 2], belief.weights)
    assert list(states[0]) == [4.0, 5.0, 0.7]
    # empty-candidate case: landmarks stay zeros
    lm2, states2 = rc.build_v5_extras(None, empty_cands=True,
                                      pos=np.asarray([4.0, 5.0]), yaw=0.7)
    assert np.all(lm2 == 0.0)
    assert list(states2[0]) == [4.0, 5.0, 0.7]


def test_assert_batch_contract_rejects_foreign_keys():
    ok = {k: None for k in rc.BASE_CONTRACT}
    rc.assert_batch_contract("v4", ok)  # must not raise
    bad = dict(ok)
    bad["target"] = [60.0, 80.0]  # true target coords are never policy input
    with pytest.raises(AssertionError):
        rc.assert_batch_contract("v4", bad)
    with pytest.raises(AssertionError):
        rc.assert_batch_contract("v5", ok)  # v5 needs landmarks + states
    ok5 = dict(ok, landmarks=None, states=None)
    rc.assert_batch_contract("v5", ok5)


def test_episode_dirs_cover_models():
    dirs = rc.episode_dirs("v4")
    assert dirs and all(d.exists() for d in dirs)
