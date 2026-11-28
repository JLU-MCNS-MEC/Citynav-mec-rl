#!/usr/bin/env python3
"""Static check: cosine continuation 5e-6 -> 5e-7 over 18 epochs x 9804 updates."""
import json, sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
STEPS, EPOCHS = 9804, 18
p=torch.nn.Parameter(torch.zeros(1)); opt=torch.optim.AdamW([p], lr=5e-6)
s=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS*EPOCHS, eta_min=5e-7)
lrs=[]
for _ in range(STEPS*EPOCHS):
    lrs.append(s.get_last_lr()[0]); opt.step(); s.step()
mono=all(b<=a+1e-15 for a,b in zip(lrs,lrs[1:]))
out={"steps":STEPS*EPOCHS,"lr_first":lrs[0],"lr_last":lrs[-1],
     "monotone":mono,"pass":abs(lrs[0]-5e-6)<1e-12 and abs(lrs[-1]-5e-7)<1e-10 and mono}
(HERE/"lr_schedule_check.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2)); sys.exit(0 if out["pass"] else 1)
