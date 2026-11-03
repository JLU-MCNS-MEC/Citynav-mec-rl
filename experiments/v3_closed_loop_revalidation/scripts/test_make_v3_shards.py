"""Tests for make_v3_shards.py: deterministic stride-3 sharding of the
corrected-protocol val manifest (same 5324 tasks V4/V5 ran on)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from make_v3_shards import make_shards

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MANIFEST = (REPO / "experiments/06_closed_loop_success_rate/results/seed42"
            / "manifest.json")


@pytest.fixture(scope="module")
def manifest_ids() -> list[str]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return [t["episode_id"] for t in manifest["tasks"]]


def test_shards_partition_all_tasks(tmp_path, manifest_ids):
    """Union of shards == manifest tasks, pairwise disjoint."""
    paths = make_shards(MANIFEST, tmp_path)
    assert [p.name for p in paths] == ["val_s0.txt", "val_s1.txt", "val_s2.txt"]
    shard_sets = [
        set(p.read_text(encoding="utf-8").splitlines()) for p in paths]
    union = set().union(*shard_sets)
    assert union == set(manifest_ids)
    assert not (shard_sets[0] & shard_sets[1])
    assert not (shard_sets[1] & shard_sets[2])
    assert not (shard_sets[0] & shard_sets[2])


def test_shard_sizes_match_existing_v4v5_layout(tmp_path, manifest_ids):
    """1775 / 1775 / 1774, stride-3 in manifest order."""
    paths = make_shards(MANIFEST, tmp_path)
    sizes = [len(p.read_text(encoding="utf-8").splitlines()) for p in paths]
    assert sizes == [1775, 1775, 1774]


def test_shard_order_is_manifest_stride(tmp_path, manifest_ids):
    """Shard i holds manifest tasks i, i+3, i+6, ... (deterministic)."""
    paths = make_shards(MANIFEST, tmp_path)
    lines0 = paths[0].read_text(encoding="utf-8").splitlines()
    assert lines0 == manifest_ids[0::3]
    lines1 = paths[1].read_text(encoding="utf-8").splitlines()
    assert lines1 == manifest_ids[1::3]
    lines2 = paths[2].read_text(encoding="utf-8").splitlines()
    assert lines2 == manifest_ids[2::3]
