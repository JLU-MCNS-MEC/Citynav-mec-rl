"""Guard the repository boundary that previously resolved to the home directory."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_root_is_the_project_directory() -> None:
    try:
        top_level = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        git_dir = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--absolute-git-dir"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        pytest.skip("not a git checkout (for example, a source archive)")
    assert Path(top_level).resolve() == PROJECT_ROOT
    assert Path(git_dir).resolve() == (PROJECT_ROOT / ".git").resolve()


def test_example_config_contains_no_server_absolute_paths() -> None:
    text = (PROJECT_ROOT / "configs" / "data_config.example.yaml").read_text(
        encoding="utf-8"
    )
    assert "/home/" not in text
    assert "/mnt/" not in text
    assert "${CITYNAV_DATA_ROOT}" in text
    assert "${CITYNAV_PLY_ROOT}" in text
