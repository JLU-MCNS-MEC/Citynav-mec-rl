#!/usr/bin/env python3
"""Write results/MANIFEST.json: size + SHA-256 for every 03B artefact.

The large files (``*.npz`` labels, ``results_naive_15deg/``) stay on the server;
the manifest lets both sides be checked against each other.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOTS = [HERE / "results", HERE / "results_naive_15deg", HERE / "figures", HERE / "cache"]
COMMITTED = ("results/teacher_", "results/summary", "results/analysis/", "figures/", "cache/obstacles/")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    entries = []
    for root in ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(HERE).as_posix()
            keep = any(rel.startswith(prefix) for prefix in COMMITTED)
            if rel.endswith(".npz") and rel.startswith("results/teacher_"):
                keep = False          # labels stay on the server
            if rel.startswith("results_naive_15deg/"):
                keep = False          # defect evidence, server only
            entries.append({"path": rel, "bytes": path.stat().st_size,
                            "sha256": sha256(path), "in_git": keep})
    manifest = {"experiment": "03b_target_directed_teacher",
                "generated_from": "results/run_teacher.py (25 deg deadband)",
                "files": entries}
    out = HERE / "results/MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    total = sum(e["bytes"] for e in entries)
    kept = sum(e["bytes"] for e in entries if e["in_git"])
    print(f"{len(entries)} files, {total / 1e6:.1f} MB total, {kept / 1e6:.1f} MB in git")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
