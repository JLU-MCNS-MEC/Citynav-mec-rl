#!/usr/bin/env python3
"""Materialise the four frozen deployments (10 nodes per map).

Writes results/protocol/deployment_<group>.json plus a combined table and
prints a compact audit (model counts, tier counts, model/tier means).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P  # noqa: E402


def main() -> int:
    P.PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    combined = {}
    for group in P.GROUPS:
        dep = P.deployment(group)
        combined[group] = dep
        (P.PROTOCOL_DIR / f"deployment_{group}.json").write_text(
            json.dumps(dep, indent=1), encoding="utf-8")
        models = Counter()
        tiers = Counter()
        per_model_tiers = defaultdict(list)
        for nodes in dep.values():
            assert len(nodes) == 10
            for n in nodes:
                models[n["model_type"]] += 1
                tiers[n["compute_tflops"]] += 1
                per_model_tiers[n["model_type"]].append(n["compute_tflops"])
        print(f"[{group}] {P.GROUP_SPECS[group]['label']}")
        print(f"  maps={len(dep)} nodes={sum(len(v) for v in dep.values())}")
        print(f"  models={dict(models)}")
        print(f"  tiers={dict(sorted(tiers.items()))}")
        for m in P.MODEL_ORDER:
            if m in per_model_tiers:
                ts = per_model_tiers[m]
                print(f"  {P.MODEL_LABEL[m]:13s} tiers={ts} "
                      f"mean={sum(ts)/len(ts):.2f}")
        # invariants
        assert len({n["model_type"] for nodes in dep.values()
                    for n in nodes}) == (
            2 if group.startswith("two") else 5)
    (P.PROTOCOL_DIR / "deployments.json").write_text(
        json.dumps(combined, indent=1), encoding="utf-8")
    print(f"wrote {P.PROTOCOL_DIR/'deployments.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
