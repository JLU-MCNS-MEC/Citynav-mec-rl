"""Path isolation helpers.

All writes must go through :func:`ensure_output_dir`, which refuses paths
outside the project.  Raw CityNav / CityRefer / raster / PLY directories are
only ever read.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

from src.config import DataConfig, PROJECT_ROOT


DATA_CACHE_ROOT = PROJECT_ROOT / "data_cache"
RUNS_ROOT = PROJECT_ROOT / "runs"
FIGURES_ROOT = PROJECT_ROOT / "figures"
REPORTS_ROOT = PROJECT_ROOT / "reports"


def _resolve(path: Path | str) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def is_within(path: Path | str, root: Path | str) -> bool:
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def ensure_output_dir(
    path: Path | str,
    *,
    data_config: DataConfig | None = None,
) -> Path:
    """Create a directory only if it is inside this project and outside raw data."""
    resolved = _resolve(path)
    if not is_within(resolved, PROJECT_ROOT):
        raise ValueError(f"refusing to write outside project: {resolved}")
    if data_config is not None:
        for raw_root in (
            data_config.data_root,
            data_config.ply_root,
        ):
            if is_within(resolved, raw_root):
                raise ValueError(
                    f"refusing to write into shared raw data: {resolved}"
                )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_output_file(
    path: Path | str,
    *,
    data_config: DataConfig | None = None,
) -> Path:
    resolved = _resolve(path)
    ensure_output_dir(resolved.parent, data_config=data_config)
    if data_config is not None:
        for raw_root in (data_config.data_root, data_config.ply_root):
            if is_within(resolved, raw_root):
                raise ValueError(
                    f"refusing to write into shared raw data: {resolved}"
                )
    return resolved


def relative_to_project(path: Path | str) -> str:
    resolved = _resolve(path)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return "<outside-project>"


def display_path(path: Path | str, data_config: DataConfig) -> str:
    """Redact server absolute paths for reports."""
    resolved = _resolve(path)
    if is_within(resolved, data_config.data_root):
        return "<CITYNAV_DATA_ROOT>/" + resolved.relative_to(
            data_config.data_root
        ).as_posix()
    if is_within(resolved, data_config.ply_root):
        return "<CITYNAV_PLY_ROOT>/" + resolved.relative_to(
            data_config.ply_root
        ).as_posix()
    if is_within(resolved, PROJECT_ROOT):
        return relative_to_project(resolved)
    return "<outside-project>"


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in sorted(filenames):
            yield Path(dirpath) / filename


def snapshot_raw_dirs(data_config: DataConfig) -> dict:
    """Cheap immutable metadata snapshot used to prove raw data was not touched."""
    roots = {
        "original_citynav": data_config.citynav_dir,
        "cityrefer": data_config.cityrefer_dir,
        "rgbd": data_config.rgbd_dir,
        "ply": data_config.ply_root,
    }
    summary: dict = {"roots": {}, "file_count": 0, "total_bytes": 0}
    digest = hashlib.sha1()
    for label, root in roots.items():
        count = 0
        total = 0
        max_mtime_ns = 0
        for path in _iter_files(root):
            try:
                stat = path.stat()
            except OSError:
                continue
            rel = path.relative_to(root).as_posix()
            count += 1
            total += int(stat.st_size)
            max_mtime_ns = max(max_mtime_ns, int(stat.st_mtime_ns))
            digest.update(
                f"{label}|{rel}|{stat.st_size}|{stat.st_mtime_ns}\n".encode()
            )
        summary["roots"][label] = {
            "exists": root.exists(),
            "file_count": count,
            "total_bytes": total,
            "max_mtime_ns": max_mtime_ns,
        }
        summary["file_count"] += count
        summary["total_bytes"] += total
    summary["digest"] = digest.hexdigest()
    return summary


def assert_snapshot_unchanged(before: dict, after: dict) -> None:
    if before.get("digest") != after.get("digest"):
        raise RuntimeError(
            "shared raw-data metadata changed during the run: "
            f"before={before.get('digest')} after={after.get('digest')}"
        )
