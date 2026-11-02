#!/usr/bin/env python3
"""Compare M-v4 (runs_11b) vs M-v5 (runs_11c) per epoch, side by side.

Prints the same-epoch rows of both runs and marks which model wins on each
metric at the epoch level, plus a best-so-far summary per model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/mnt/hdd2t/workspaces/Citynav-mec-rl-fixed50m")
B = REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4/results/runs_11b/seed42/history.jsonl"
C = REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4/results/runs_11c/seed42/history.jsonl"


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def fmt(d: dict | None, key: str, nd: int = 4) -> str:
    if d is None:
        return "-".rjust(8)
    v = d["val_seen"]
    val = {"f1": v["macro_f1"], "acc": v["accuracy"],
           "stop_r": v["per_class"]["Stop"]["recall"],
           "tl_r": v["per_class"]["TurnLeft"]["recall"],
           "tr_r": v["per_class"]["TurnRight"]["recall"],
           "loss": d["train_loss"]}.get(key)
    return f"{val:.{nd}f}".rjust(8)


def main() -> int:
    b, c = load(B), load(C)
    if not c:
        print("v5 run not started yet")
        return 0
    n = min(len(b), len(c))
    print(f"epoch |  v4 F1  v5 F1   | v4 acc v5 acc  | v4Stop v5Stop | v4TL   v5TL   | v4TR   v5TR")
    for e in range(n):
        d4, d5 = b[e], c[e]
        v4f, v5f = d4["val_seen"]["macro_f1"], d5["val_seen"]["macro_f1"]
        win = "<" if v4f > v5f else (">" if v5f > v4f else "=")
        print(f"{e:5d} | {v4f:.4f} {v5f:.4f} {win} | {d4['val_seen']['accuracy']:.4f} {d5['val_seen']['accuracy']:.4f} | "
              f"{d4['val_seen']['per_class']['Stop']['recall']:.3f} {d5['val_seen']['per_class']['Stop']['recall']:.3f} | "
              f"{d4['val_seen']['per_class']['TurnLeft']['recall']:.3f} {d5['val_seen']['per_class']['TurnLeft']['recall']:.3f} | "
              f"{d4['val_seen']['per_class']['TurnRight']['recall']:.3f} {d5['val_seen']['per_class']['TurnRight']['recall']:.3f}")
    def best(rows: list[dict]) -> dict:
        if not rows:
            return {}
        i = max(range(len(rows)), key=lambda k: rows[k]["val_seen"]["macro_f1"])
        return {"epoch": rows[i]["epoch"],
                "f1": rows[i]["val_seen"]["macro_f1"],
                "stop_r": rows[i]["val_seen"]["per_class"]["Stop"]["recall"],
                "tl_r": rows[i]["val_seen"]["per_class"]["TurnLeft"]["recall"],
                "tr_r": rows[i]["val_seen"]["per_class"]["TurnRight"]["recall"]}
    bb, bc = best(b), best(c)
    print(f"\nbest so far:  v4 epoch {bb.get('epoch', '-')} F1={bb.get('f1', float('nan')):.4f} | "
          f"v5 epoch {bc.get('epoch', '-')} F1={bc.get('f1', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
