#!/usr/bin/env python3
"""Contract tests for Stage 08 (anti-cheat, controller rules, determinism)."""

from __future__ import annotations

import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "06_closed_loop_success_rate/scripts"))
sys.path.insert(0, str(ROOT / "05_transformer_m_training/scripts"))
sys.path.insert(0, str(ROOT / "04_visual_goal_global_map_pipeline/scripts"))

import geometry_controller  # noqa: E402
import geometry_env  # noqa: E402
from geometry_controller import ALIGN_DEG, STOP_RADIUS_M, Top1Geometry, Top5WeightedGeometry  # noqa: E402

STAGE06 = ROOT / "06_closed_loop_success_rate/results/seed42"


def _belief(**kw) -> np.ndarray:
    b = np.zeros(12, np.float32)
    b[11] = 1.0
    for k, v in kw.items():
        b[int(k[1:])] = v
    return b


def test_controller_only_accepts_a_belief_vector():
    c = Top1Geometry()
    sig = inspect.signature(c.act)
    assert list(sig.parameters) == ["belief"]
    assert list(inspect.signature(c.diagnostics).parameters) == ["belief"]
    for bad in (np.zeros(11, np.float32), np.zeros((1, 12), np.float32)):
        try:
            c.act(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("controller accepted a non-(12,) input")


def _ast_names_modules(path: Path):
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names, modules = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names, modules


def test_controller_module_has_no_task_or_target_access():
    """The controller module may only import math/numpy and touch the belief."""
    names, modules = _ast_names_modules(HERE / "geometry_controller.py")
    assert modules <= {"__future__", "math", "numpy"}, modules
    assert not (names & {"task", "target", "episode_id", "manifest", "heading_crop",
                         "TransformerM"}), names & {"task", "target", "episode_id",
                                                    "manifest", "heading_crop", "TransformerM"}


def test_decision_rules():
    c = Top1Geometry()
    # estimated distance <= 20 -> STOP, regardless of bearing
    assert c.act(_belief(b3=100, b4=100, b5=20.0)) == 0
    # 25 deg is the alignment threshold (strict)
    b = _belief(b3=100.0, b4=100.0 * math.tan(math.radians(ALIGN_DEG + 1)), b5=100)
    assert c.act(b) == 2
    b = _belief(b3=100.0, b4=-100.0 * math.tan(math.radians(ALIGN_DEG + 1)), b5=100)
    assert c.act(b) == 3
    b = _belief(b3=100.0, b4=100.0 * math.tan(math.radians(ALIGN_DEG - 1)), b5=100)
    assert c.act(b) == 1
    assert STOP_RADIUS_M == 20.0 and ALIGN_DEG == 25.0


def test_top5_uses_the_weighted_components():
    c = Top5WeightedGeometry()
    assert c.act(_belief(b6=100, b7=0, b8=15, b3=0, b4=0, b5=500)) == 0
    assert c.act(_belief(b6=100, b7=100, b8=100, b3=0, b4=0, b5=1)) == 2


def test_env_never_encodes_the_current_view_or_uses_a_transformer():
    names, modules = _ast_names_modules(HERE / "geometry_env.py")
    assert not (names & {"heading_crop", "TransformerM", "history_block", "torch"})
    assert modules <= {"__future__", "math", "sys", "time", "pathlib", "numpy",
                       "closed_loop", "dataset_v2", "pipeline", "geometry_controller"}
    # the encoder is used only for the goal image
    src = (HERE / "geometry_env.py").read_text(encoding="utf-8")
    assert "encoder.embed([raster.north_crop" in src


def test_target_embedding_matches_the_training_cache():
    """The goal image embedding must be the one Stage 05 cached."""
    import dataset_v2
    from geometry_env import target_vector

    store = dataset_v2.EpisodeStore("val_seen", dataset_v2.CACHE)
    eps = dataset_v2.load_split("val_seen")
    pos = {e.key: i for i, e in enumerate(eps)}
    manifest = json.loads((STAGE06 / "manifest.json").read_text(encoding="utf-8"))
    task = next(t for t in manifest["tasks"] if t["split"] == "val_seen")
    e = pos[task["index"]]
    encoder = dataset_v2.FrozenEncoder("cuda")
    rasters = dataset_v2.RasterStore()
    vec = target_vector(encoder, rasters.get(task["map"]), task)
    cached = store.array(e)["target_emb"]
    assert cached.dtype == np.float16
    assert np.array_equal(vec.astype(np.float16), cached)


def test_episode_is_deterministic():
    import dataset_v2
    from geometry_controller import make
    from geometry_env import run_episode

    manifest = json.loads((STAGE06 / "manifest.json").read_text(encoding="utf-8"))
    task = dict(next(t for t in manifest["tasks"] if t["split"] == "val_seen"), manifest_sha="x")
    encoder = dataset_v2.FrozenEncoder("cuda")
    rasters, cands = dataset_v2.RasterStore(), dataset_v2.CandidateStore()
    a = run_episode(task, make("top1"), encoder, rasters, cands, 200)
    b = run_episode(task, make("top1"), encoder, rasters, cands, 200)
    assert a["trajectory"] == b["trajectory"]
    assert a["success"] == b["success"] and a["steps"] == b["steps"]


def test_smoke_manifest_lists_twelve_cells(tmp_path=None):
    smoke = HERE.parent / "results/smoke_ids.txt"
    if not smoke.exists():
        print("smoke ids missing, skipped")
        return
    ids = [x.strip() for x in smoke.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(ids) == 12 and len(set(ids)) == 12
    manifest = json.loads((STAGE06 / "manifest.json").read_text(encoding="utf-8"))
    by_id = {t["episode_id"]: t for t in manifest["tasks"]}
    cells = {(by_id[i]["split"], by_id[i]["length_group"]) for i in ids}
    assert len(cells) == 6
    assert all(sum(1 for i in ids if (by_id[i]["split"], by_id[i]["length_group"]) == c) == 2
               for c in cells)
