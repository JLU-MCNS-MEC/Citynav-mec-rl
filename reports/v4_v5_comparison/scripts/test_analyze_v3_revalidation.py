"""Tests for analyze_v3_revalidation.py on synthetic journals (no GPU/data)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from analyze_v3_revalidation import load, per_episode_alignment, stats


def _write_journal(dir_path: Path, prefix: str, shard: int, recs: list[dict]) -> Path:
    d = dir_path / f"{prefix}_s{shard}"
    d.mkdir(parents=True, exist_ok=True)
    jf = d / "journal.jsonl"
    with jf.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return dir_path


def _rec(eid: str, split: str, success: bool, final_dist: float,
         steps: int, failure: str) -> dict:
    return {"episode_id": eid, "split": split, "map": "birmingham_block_0",
            "success": success, "final_dist": final_dist, "steps": steps,
            "spl": 0.8 if success else 0.0, "osr": success, "failure": failure,
            "execution_status": "ok", "length_group": "short"}


def test_load_merges_shards(tmp_path):
    _write_journal(tmp_path, "v3_val", 0, [_rec("val_seen:00000", "val_seen", True, 10.0, 5, "success")])
    _write_journal(tmp_path, "v3_val", 1, [_rec("val_seen:00003", "val_seen", False, 40.0, 9, "never_stop")])
    recs = load(tmp_path, "v3_val")
    assert set(recs) == {"val_seen:00000", "val_seen:00003"}


def test_stats_computes_sr_ne_spl_failures(tmp_path):
    _write_journal(tmp_path, "v3_val", 0, [
        _rec("val_seen:00000", "val_seen", True, 10.0, 5, "success"),
        _rec("val_seen:00003", "val_seen", False, 40.0, 9, "never_stop"),
        _rec("val_unseen:00006", "val_unseen", False, 90.0, 200, "premature_stop"),
    ])
    s = stats(load(tmp_path, "v3_val"), "v3")
    assert s["n_evalable"] == 3
    assert abs(s["sr"] - 1 / 3) < 1e-9
    assert abs(s["osr"] - 1 / 3) < 1e-9
    assert abs(s["ne_mean"] - (10 + 40 + 90) / 3) < 1e-9
    assert s["spl"] == 0.8 / 3
    assert s["failures"] == {"never_stop": 1, "premature_stop": 1}
    assert s["by_split"]["val_seen"]["sr"] == 0.5
    assert s["by_split"]["val_unseen"]["sr"] == 0.0


def test_per_episode_alignment_counts_and_csv(tmp_path):
    v4 = {"val_seen:00000": _rec("val_seen:00000", "val_seen", True, 10.0, 5, "success"),
          "val_seen:00003": _rec("val_seen:00003", "val_seen", True, 12.0, 6, "success"),
          "val_seen:00006": _rec("val_seen:00006", "val_seen", False, 80.0, 40, "out_of_bounds")}
    v3 = {"val_seen:00003": _rec("val_seen:00003", "val_seen", False, 45.0, 20, "never_stop"),
          "val_seen:00006": _rec("val_seen:00006", "val_seen", False, 90.0, 30, "premature_stop"),
          "val_seen:00009": _rec("val_seen:00009", "val_seen", True, 8.0, 4, "success")}
    out_csv = tmp_path / "align.csv"
    align = per_episode_alignment(v4, v3, out_csv)
    assert align["common"] == 2
    assert align["v4_only_success"] == 1          # 00003
    assert align["v3_only_success"] == 0          # no common eid where v3 wins
    assert align["neither"] == 1                  # 00006
    rows = list(csv.DictReader(out_csv.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == {"episode_id": "val_seen:00003", "split": "val_seen",
                       "map": "birmingham_block_0", "v4_success": "True",
                       "v3_success": "False", "v4_final_dist": "12.0",
                       "v3_final_dist": "45.0", "v4_steps": "6", "v3_steps": "20",
                       "v4_spl": "0.8000", "v3_spl": "0.0000"}
    assert [r["episode_id"] for r in rows] == ["val_seen:00003", "val_seen:00006"]
