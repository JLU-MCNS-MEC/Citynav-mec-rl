"""Tests for verify_videos.py checks on synthetic inputs."""
from __future__ import annotations

from pathlib import Path

import pytest

from verify_videos import (audit_contracts, check_metadata, check_replay,
                           check_video_props)

HERE = Path(__file__).resolve().parent


def test_check_video_props_flags_wrong_codec_resolution_frames():
    good = {"streams": [{"codec_type": "video", "codec_name": "h264",
                         "width": 1280, "height": 720, "pix_fmt": "yuv420p"}],
            "format": {"duration": "10.5"}}
    assert check_video_props(good, 1280, 720, 105) == []
    bad = {"streams": [{"codec_type": "video", "codec_name": "vp9",
                        "width": 640, "height": 360, "pix_fmt": "yuv420p"}],
           "format": {"duration": "3.0"}}
    errs = check_video_props(bad, 1280, 720, 105)
    assert any("codec" in e for e in errs)
    assert any("640x360" in e for e in errs)


def test_check_replay_flags_inconsistencies():
    good = {"argmax_fully_consistent": True, "prob_sums_all_one": True,
            "steps": 10, "checkpoint_sha256": "abc", "trajectory": [{}] * 10}
    assert check_replay(good, "abc") == []
    errs = check_replay(good, "def")
    assert any("checkpoint sha" in e for e in errs)
    bad = {"argmax_fully_consistent": False, "prob_sums_all_one": False,
           "steps": 10, "checkpoint_sha256": "abc", "trajectory": [{}] * 9}
    errs = check_replay(bad, "abc")
    assert any("argmax" in e for e in errs)
    assert any("prob sum" in e for e in errs)
    assert any("length" in e for e in errs)


def test_check_metadata_validates_refs(tmp_path):
    meta = {"episode_id": "test_unseen:00000",
            "models": {"v5": {"trajectory_ref": str(tmp_path / "missing.json"),
                              "trajectory_sha256": "x",
                              "replay_ref": str(tmp_path / "missing_r.json"),
                              "replay_sha256": "y",
                              "replay_argmax_consistent": True}}}
    errs = check_metadata(meta)
    assert any("missing" in e for e in errs)
    # now with a real file whose sha is recomputable
    src = tmp_path / "ep.json"
    src.write_text('{"a": 1}')
    import hashlib
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    meta["models"]["v5"]["trajectory_ref"] = str(src)
    meta["models"]["v5"]["trajectory_sha256"] = sha
    errs = check_metadata(meta)
    assert not any("trajectory" in e for e in errs)


def test_audit_contracts_finds_runtime_guard_in_replay_source():
    """The replay script must enforce the input contract at call time, and
    its contract must contain no true-target key."""
    src = (HERE / "replay_corrected.py").read_text(encoding="utf-8")
    errs = audit_contracts(src)
    assert errs == []
    poisoned = src.replace('BASE_CONTRACT = ["current"',
                           'BASE_CONTRACT = ["target", "current"')
    assert any("target" in e for e in audit_contracts(poisoned))
    assert audit_contracts("")  # missing guard must be flagged
