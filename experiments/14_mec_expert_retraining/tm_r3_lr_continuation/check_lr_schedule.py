#!/usr/bin/env python3
"""Static check: cosine continuation LR 5e-6 -> 5e-7, no warmup, monotone."""
import json, math, sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
STEPS = 19608          # updates per epoch measured in Stage 14 TM
EPOCHS = 18            # epochs 2..19
TOTAL = STEPS * EPOCHS

p = torch.nn.Parameter(torch.zeros(1))
opt = torch.optim.AdamW([p], lr=5e-6)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TOTAL, eta_min=5e-7)
lrs = []
for _ in range(TOTAL):
    lrs.append(sch.get_last_lr()[0]); opt.step(); sch.step()
first, last, mn = lrs[0], lrs[-1], min(lrs)
monotone = all(b <= a + 1e-15 for a, b in zip(lrs, lrs[1:]))
res = {"steps": TOTAL, "lr_first": first, "lr_last": last, "lr_min": mn,
       "monotone_nonincreasing": monotone,
       "no_step_above_start": max(lrs) <= 5e-6 + 1e-15,
       "pass": abs(first - 5e-6) < 1e-12 and abs(last - 5e-7) < 1e-10 and monotone}
(HERE / "lr_schedule_check.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
sys.exit(0 if res["pass"] else 1)
