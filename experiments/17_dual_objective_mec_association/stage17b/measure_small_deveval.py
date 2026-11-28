#!/usr/bin/env python3
"""Recompute the Stage-11 SmallMEC4 Train-internal Dev Macro-F1 (no retraining).

The historical 0.7243 came from Val-Seen model selection; this script measures
the same checkpoint on the Stage-14 Train-internal Dev split so the Stage-17
V2 quality prior is comparable with the other experts.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
for p in [REPO / "experiments/14_mec_expert_retraining",
          REPO / "experiments/14_mec_expert_retraining/stability_revision",
          REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4/scripts",
          REPO / "experiments/05_transformer_m_training/scripts"]:
    sys.path.insert(0, str(p))

import train_stable as TS
from dataset11b import ChunkedMemmapDataset
from model11_small import SmallMEC4
from train_stage14 import collate11, to_model_batch11

CKPT = REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4/results/runs_small/seed42/best.pt"

def main():
    dev = torch.device("cuda")
    ep = TS.load_ep_ranges()
    dev_eps = TS.make_dev_split(ep, 42, 0.02)
    dev_idx = TS.split_states(ep, dev_eps)
    ds = ChunkedMemmapDataset("train_seen", dev_idx, blank_mask=None)
    loader = DataLoader(ds, batch_size=128, num_workers=8, collate_fn=collate11,
                        pin_memory=True)
    m = SmallMEC4().to(dev).eval()
    m.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"])
    preds, labels = [], []
    with torch.no_grad():
        for b in loader:
            logits = m(to_model_batch11(b, None, dev))
            preds.append(logits.argmax(-1).cpu().numpy())
            labels.append(b["label"])
    p = np.concatenate(preds); y = np.concatenate(labels)
    f1s, per = [], {}
    for c, name in enumerate(["Stop", "Forward", "TurnLeft", "TurnRight"]):
        tp = int(((p == c) & (y == c)).sum()); fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        f = 2 * pr * rc / max(pr + rc, 1e-9)
        f1s.append(f); per[name] = {"precision": pr, "recall": rc, "f1": f,
                                    "support": int((y == c).sum())}
    out = {"checkpoint": str(CKPT), "dev_states": int(len(y)),
           "macro_f1": float(np.mean(f1s)), "accuracy": float((p == y).mean()),
           "per_class": per,
           "note": "Train-internal Dev (seed42, dev_frac=0.02), no normalisation (Small)"}
    (HERE / "small_deveval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("dev_states", "macro_f1", "accuracy")}, indent=2))
    print(json.dumps(per, indent=2))

if __name__ == "__main__":
    main()
