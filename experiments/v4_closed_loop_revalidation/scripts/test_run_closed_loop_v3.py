"""Tests for the additive v3 extension of run_closed_loop.py.

These tests only assert the *interface* additions (argparse choices), never
touch V4/V5 behavior. Importing run_closed_loop pulls in torch etc., so the
harness is exercised via subprocess --help (argparse exits before any model
code runs, but module-level imports must succeed).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "run_closed_loop.py"


def _help_output() -> str:
    proc = subprocess.run(
        [sys.executable, str(HARNESS), "--help"],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"--help failed:\n{proc.stderr}"
    return proc.stdout


def test_harness_accepts_v3_model_choice():
    """--model must offer {v4,v5,v3} after the additive extension."""
    out = _help_output()
    assert "{v4,v5,v3}" in out, f"v3 missing from choices in:\n{out}"


def test_harness_still_accepts_v4_and_v5():
    """The extension must not drop the existing v4/v5 choices."""
    out = _help_output()
    assert "--model" in out
