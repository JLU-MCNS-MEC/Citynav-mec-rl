#!/usr/bin/env python3
"""Stage 00: archive the token-v1 artefacts and delete generated MEC token data.

Dry run (default) inventories everything and writes the manifests.  ``--execute``
then deletes only the enumerated token payloads, rescans, and records the freed
disk space.  It never touches raw CityNav JSON/PLY/rasters, source code,
checkpoints, experiment reports, or the archived copy.

Usage:
    python archive_and_cleanup.py --dry-run
    python archive_and_cleanup.py --execute
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/20261024_1/Workspace/Citynav-mec-rl")
ARCHIVE = Path("/home/20261024_1/Workspace/Citynav-archive/20261029_token_v1")
AUDIT = REPO / "experiments/visual_encoder_citynav/token_alignment_audit_20261029/cache"
TOKENS = REPO / "data_cache/t5_frozen_1000/tokens"
AUDIT_CACHES = (
    "probe_h0",
    "probe_h0_pretrain",
    "probe_h16",
    "subset_h0",
    "subset_h0_pretrain",
    "subset_h16",
)
REPORT_SNAPSHOTS = (
    "experiments/STATUS_2026-10-29.md",
    "experiments/README.md",
    "experiments/02_full_closed_loop_recheck/REPORT.md",
    "experiments/03_v82_citynav_dataflow_audit/REPORT.md",
    "experiments/06_token_sampling_design_open_issue/OPEN_ISSUE.md",
    "experiments/visual_encoder_citynav/REPORT.md",
    "experiments/visual_encoder_citynav/STATUS.md",
    "experiments/visual_encoder_citynav/ablations/20261028/REPORT.md",
    "experiments/visual_encoder_citynav/ablations/20261028/REPORT_BEV_POSE.md",
    "experiments/visual_encoder_citynav/e2_fov_pose_extended_seed42/REPORT_EXT5502.md",
    "experiments/visual_encoder_citynav/token_alignment_audit_20261029/REPORT.md",
    "experiments/visual_encoder_citynav/frozen_e2_fov_pose_mec/STATUS.md",
    "experiments/visual_encoder_citynav/frozen_e2_fov_pose_mec/closed_loop/formal/CLOSED_LOOP_REPORT.md",
    "experiments/visual_encoder_citynav/frozen_e2_fov_pose_mec/closed_loop/sqrt/SQRT_CLOSED_LOOP_REPORT.md",
)


def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _disk_bytes(path: Path) -> int:
    st = path.stat()
    return int(st.st_blocks * 512)


def _tree_size(root: Path) -> tuple[int, int, int]:
    """Return (files, apparent bytes, disk bytes) for a file or directory."""
    if not root.exists():
        return 0, 0, 0
    if root.is_file():
        return 1, root.stat().st_size, _disk_bytes(root)
    files = apparent = disk = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            try:
                files += 1
                apparent += p.stat().st_size
                disk += _disk_bytes(p)
            except FileNotFoundError:
                continue
    return files, apparent, disk


def _sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _targets() -> list[dict]:
    targets = [
        {"path": TOKENS / "t5_tokens.npz", "kind": "core token array"},
        {"path": TOKENS / "t5_tokens.json", "kind": "core token metadata"},
        {"path": TOKENS / "shards", "kind": "core token shards"},
    ]
    for name in AUDIT_CACHES:
        base = AUDIT / name
        targets.append({"path": base / "tokens/t5_tokens.npz", "kind": "audit token array"})
        targets.append({"path": base / "tokens/t5_tokens.json", "kind": "audit token metadata"})
        parts = base / "parts"
        if parts.exists():
            for p in sorted(parts.glob("*.npz")):
                targets.append({"path": p, "kind": "audit retokenized part"})
    return targets


def _scan_token_dirs() -> list[dict]:
    """Report any other generated token dataset directory we can identify."""
    rows = []
    for root in (Path("/home/20261024_1/Workspace"),):
        for path in root.rglob("token_dataset_*"):
            if path.is_dir() and "site-packages" not in str(path):
                files, apparent, disk = _tree_size(path)
                rows.append(
                    {
                        "path": str(path),
                        "files": files,
                        "apparent_bytes": apparent,
                        "disk_bytes": disk,
                    }
                )
    return rows


def _weight_files() -> list[Path]:
    files: list[Path] = []
    for p in sorted((REPO / "runs/visual_encoder_citynav").glob("*/best_encoder.pt")):
        files.append(p)
    for p in (
        REPO / "data_cache/weights/shared_encoder.pt",
        REPO / "data_cache/t5_frozen_1000/retrieval/encoder.pt",
        REPO / "data_cache/t5_frozen_1000/retrieval/retriever.pt",
    ):
        if p.exists():
            files.append(p)
    for sub in ("mec", "mec_balanced", "mec_sqrt"):
        for p in sorted((REPO / "data_cache/t5_frozen_1000" / sub).glob("*/mec.pt")):
            files.append(p)
    return files


def _token_stats() -> dict:
    meta_path = TOKENS / "t5_tokens.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    by_split: dict[str, dict[str, int]] = {}
    maps = set()
    for frame in meta["frames_metadata"]:
        split = frame["split"]
        maps.add(frame["map_name"])
        action = str(frame["action"])
        by_split.setdefault(split, {}).setdefault(action, 0)
        by_split[split][action] += 1
    return {
        "frames": meta["frames"],
        "trajectories": meta["trajectories"],
        "action_counts": meta["action_counts"],
        "action_counts_by_split": by_split,
        "maps": sorted(maps),
        "n_maps": len(maps),
        "gab_reliability_threshold": meta["gab_reliability_threshold"],
        "gab_config": meta["gab_config"],
        "test_unseen_read": meta["test_unseen_read"],
        "token_file": meta["token_file"],
        "token_bytes": meta["token_bytes"],
    }


def _array_shapes() -> dict:
    import numpy as np

    out = {}
    with np.load(TOKENS / "t5_tokens.npz", mmap_mode="r", allow_pickle=False) as z:
        for key in z.files:
            out[key] = {"shape": list(z[key].shape), "dtype": str(z[key].dtype)}
    return out


def _df_bytes() -> int:
    st = os.statvfs("/home/20261024_1")
    return st.f_bavail * st.f_frsize


def _write_archive(inventory: list[dict], stats: dict, shapes: dict, weights: list[dict]) -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    (ARCHIVE / "metadata").mkdir(exist_ok=True)
    (ARCHIVE / "reports_snapshot").mkdir(exist_ok=True)
    shutil.copy2(TOKENS / "t5_tokens.json", ARCHIVE / "metadata/t5_tokens.json")
    (ARCHIVE / "metadata/token_dataset_stats.json").write_text(
        json.dumps({"stats": stats, "arrays": shapes}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (ARCHIVE / "metadata/token_cleanup_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (ARCHIVE / "weights_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(weights)
    with (ARCHIVE / "weights_sha256.txt").open("w", encoding="utf-8") as fh:
        for row in weights:
            fh.write(f"{row['sha256']}  {row['path']}\n")
    for rel in REPORT_SNAPSHOTS:
        src = REPO / rel
        if src.exists():
            dst = ARCHIVE / "reports_snapshot" / rel.replace("/", "__")
            shutil.copy2(src, dst)
    env = [
        f"created_at: {_now()}",
        f"source_repo: {REPO}",
        f"git_branch: {_git('branch', '--show-current')}",
        f"git_head: {_git('log', '-1', '--format=%H %ci %s')}",
        f"archive_tag: archive/citynav-token-v1-20261029 -> "
        f"{_git('rev-parse', 'archive/citynav-token-v1-20261029')}",
        f"disk_free_bytes: {_df_bytes()}",
    ]
    (ARCHIVE / "environment.txt").write_text("\n".join(env) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="delete after archiving")
    parser.add_argument("--dry-run", action="store_true", help="inventory only (default)")
    args = parser.parse_args()
    execute = args.execute and not args.dry_run

    inventory = []
    for target in _targets():
        path: Path = target["path"]
        files, apparent, disk = _tree_size(path)
        inventory.append(
            {
                "path": str(path),
                "kind": target["kind"],
                "exists": path.exists(),
                "files": files,
                "apparent_bytes": apparent,
                "disk_bytes": disk,
            }
        )

    stats = _token_stats()
    shapes = _array_shapes()
    weights = [
        {"path": str(p.relative_to(REPO)), "bytes": p.stat().st_size, "sha256": _sha256(p)}
        for p in _weight_files()
    ]
    extra = _scan_token_dirs()

    before = _df_bytes()
    _write_archive(inventory, stats, shapes, weights)

    total_apparent = sum(row["apparent_bytes"] for row in inventory)
    total_disk = sum(row["disk_bytes"] for row in inventory)
    print(f"targets: {len(inventory)}  files: {sum(r['files'] for r in inventory)}")
    print(f"apparent bytes: {total_apparent}  disk bytes: {total_disk}")
    print(f"other token_dataset_* dirs: {len(extra)}")
    print(f"weights hashed: {len(weights)}")
    print(f"archive: {ARCHIVE}")

    if not execute:
        print("dry run only; pass --execute to delete")
        return 0

    deleted = []
    for row in inventory:
        path = Path(row["path"])
        if not row["exists"]:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        deleted.append({"path": str(path), "files": row["files"], "disk_bytes": row["disk_bytes"]})
    # remove the now-empty core tokens directory
    try:
        if TOKENS.exists() and not any(TOKENS.iterdir()):
            TOKENS.rmdir()
    except OSError:
        pass

    after = _df_bytes()
    residual = []
    for row in inventory:
        path = Path(row["path"])
        if path.exists():
            residual.append(str(path))
    result = {
        "executed_at": _now(),
        "deleted": deleted,
        "apparent_bytes_deleted": total_apparent,
        "disk_bytes_deleted": total_disk,
        "df_free_before": before,
        "df_free_after": after,
        "df_freed": after - before,
        "residual_targets": residual,
        "other_token_dirs_found": extra,
    }
    (ARCHIVE / "metadata/token_cleanup_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: result[k] for k in (
        "apparent_bytes_deleted", "disk_bytes_deleted", "df_freed", "residual_targets"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
