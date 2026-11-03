"""Tests for gen_index.py index builder on synthetic metadata."""
from __future__ import annotations

import json
from pathlib import Path

from gen_index import build_index


def _meta(tmp_path: Path, name: str, cat: str, eid: str, winner: str,
          loser: str, failure: str) -> Path:
    d = tmp_path / "metadata"
    d.mkdir(exist_ok=True)
    p = d / f"{name}.json"
    p.write_text(json.dumps({
        "episode_id": eid, "category": cat, "split": "test_unseen",
        "map": "birmingham_block_0", "difficulty": "n/a",
        "winner_model": winner, "loser_model": loser,
        "models": {winner: {"success": True, "failure": "success",
                            "ckpt_sha256": "aaa111", "steps": 20},
                   loser: {"success": False, "failure": failure,
                           "ckpt_sha256": "bbb222", "steps": 12}},
        "video": {"path": f"/x/case_studies/{cat}/{name}.mp4",
                  "sha256": "cc333", "bytes": 1000},
        "conclusion": "computed conclusion"}))
    return p


def test_build_index_table_and_protocol(tmp_path):
    _meta(tmp_path, "test_unseen_00000_v5_vs_v4", "v5_success_v4_failure",
          "test_unseen:00000", "v5", "v4", "out_of_bounds")
    _meta(tmp_path, "val_unseen_00003_v4_vs_v5", "v4_success_v5_failure",
          "val_unseen:00003", "v4", "v5", "premature_stop")
    ctx = {"protocol": "corrected-history-pose-emptycand",
           "stop_radius_m": 20, "step_cap": 200,
           "v3_scope": "val-only corrected re-eval"}
    md = build_index(tmp_path / "metadata", ctx)
    assert "| Category | Episode |" in md
    assert "test_unseen:00000" in md and "v5_success_v4_failure" in md
    assert "out_of_bounds" in md and "premature_stop" in md
    assert "corrected-history-pose-emptycand" in md
    assert "20" in md and "200" in md
    assert "visualization only" in md.lower()
    assert "val-only corrected re-eval" in md
    assert ".mp4" in md and ".json" in md       # video + metadata links


def test_build_index_sorts_by_category_then_episode(tmp_path):
    _meta(tmp_path, "b_v4_vs_v5", "v4_success_v5_failure", "val_unseen:00003",
          "v4", "v5", "premature_stop")
    _meta(tmp_path, "a_v5_vs_v4", "v5_success_v4_failure", "test_unseen:00000",
          "v5", "v4", "out_of_bounds")
    md = build_index(tmp_path / "metadata", {})
    # category order dominates: A (v5 win) before C (v4 win)
    assert md.index("v5_success_v4_failure") < md.index("v4_success_v5_failure")
