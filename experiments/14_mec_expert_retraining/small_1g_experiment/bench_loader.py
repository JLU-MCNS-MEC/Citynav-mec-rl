#!/usr/bin/env python3
"""Measure data-pipeline throughput for different DataLoader settings."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
EXP14 = HERE.parent
sys.path.insert(0, str(EXP14))
for p in [EXP14.parents[0] / "experiments/05_transformer_m_training/scripts",
          EXP14.parents[0] / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4/scripts"]:
    sys.path.insert(0, str(p))
import train_stage14 as T
from dataset11b import ChunkedMemmapDataset

def bench(workers, pin, batches=200, batch=128):
    ep = T.load_ep_ranges()
    if isinstance(ep, tuple):
        ep = ep[0]
    eps = np.asarray(sorted(ep), dtype=np.int64)[:200]
    idx = np.concatenate([np.arange(ep[k][0], ep[k][1], dtype=np.int64) for k in eps])
    blank = T.build_blank_mask(42, 0, eps, ep, 0.05)
    ds = ChunkedMemmapDataset("train_seen", idx, blank_mask=blank)
    kw = dict(num_workers=workers, collate_fn=T.collate11, persistent_workers=workers > 0,
              prefetch_factor=4 if workers else None, pin_memory=pin)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, **kw)
    it = iter(loader); next(it)                      # warm-up
    t0 = time.time(); n = 0
    for _ in it:
        n += 1
        if n >= batches: break
    dt = time.time() - t0
    return {"workers": workers, "pin": pin, "batches": n,
            "batches_per_s": n / dt, "samples_per_s": n * batch / dt, "sec": dt}

if __name__ == "__main__":
    out = []
    for w, pin in ((8, False), (8, True), (16, True), (24, True)):
        r = bench(w, pin); out.append(r)
        print(json.dumps(r), flush=True)
    (HERE / "loader_bench.json").write_text(json.dumps(out, indent=2))
