#!/usr/bin/env python3
"""Stage 15 Phase B: shared-state expert action analysis.

Samples representative training states, builds the full multimodal input
(current / target / history / state / landmark / relation / masks), then runs
all five frozen experts on the SAME states and records logits, probabilities,
actions, confidence and entropy.

Models are loaded one at a time (Large is never co-resident with the others).
Small/Medium use their original (un-normalized) preprocessing; TS/TM/Large use
the Stage 14-R2 Train-only normalization, exactly as in their closed-loop evals.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent           # .../15_expert_complementarity/state_level
REPO = HERE.parents[2]
EXP14 = REPO / "experiments/14_mec_expert_retraining"
for p in [EXP14,
          REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4/scripts",
          REPO / "experiments/10_goal_query_mec_training/10D_transformer_m_training/scripts",
          REPO / "experiments/05_transformer_m_training/scripts",
          REPO / "experiments/04_visual_goal_global_map_pipeline/scripts"]:
    sys.path.insert(0, str(p))

from models import MODEL_NAMES, build_model            # noqa: E402
from dataset11b import ChunkedMemmapDataset            # noqa: E402
from train_11b import collate11, to_model_batch11      # noqa: E402

DATA = REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4/results/data/train_seen_memmap"
CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]
KEYS = ["small", "medium", "ts", "tm", "large"]
CKPTS = {
    "small": EXP14 / "checkpoints/Small/best.pt",
    "medium": EXP14 / "checkpoints/Medium/best.pt",
    "ts": EXP14 / "stability_revision/runs/ts/seed42/checkpoints/best.pt",
    "tm": EXP14 / "stability_revision/runs/tm/seed42/checkpoints/best.pt",
    "large": EXP14 / "stability_revision/runs/large/seed42/checkpoints/best.pt",
}
NORMALIZED = {"ts", "tm", "large"}   # Stage 14-R2 normalization
NORM_FILE = EXP14 / "stability_revision/normalization_statistics.json"


def load_norm() -> dict:
    d = json.loads(NORM_FILE.read_text())["standardized"]
    return {k: (np.asarray(d[k]["mean"], np.float32),
                np.asarray(d[k]["std"], np.float32))
            for k in ("belief", "match", "history_pose")}


def apply_norm(batch: dict, norm: dict) -> dict:
    for k, (mean, std) in norm.items():
        if k in batch:
            batch[k] = ((batch[k] - mean) / std).astype(np.float32)
    return batch


def stratified_indices(n_total: int, seed: int) -> np.ndarray:
    """Sample by label with oversampling of the rare Stop class."""
    labels = np.load(DATA / "labels.npy", mmap_mode="r")
    hm = np.load(DATA / "hist_mask.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    target = {0: 150, 1: 450, 2: 200, 3: 200}   # 1000 total
    chosen = []
    for c, want in target.items():
        idx = np.flatnonzero(np.asarray(labels) == c)
        if len(idx) > want:
            idx = rng.choice(idx, size=want, replace=False)
        chosen.append(idx)
    return np.sort(np.concatenate(chosen)).astype(np.int64)


def build_batches(indices: np.ndarray) -> list[dict]:
    """Preload full multimodal samples (batch=1 dicts) into RAM."""
    ds = ChunkedMemmapDataset("train_seen", indices, blank_mask=None)
    out = []
    for i in range(len(ds)):
        s = ds[i]
        out.append({k: (np.expand_dims(v, 0) if isinstance(v, np.ndarray) else v)
                    for k, v in s.items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    indices = stratified_indices(1000, args.seed)
    if args.limit:
        indices = indices[: args.limit]
    batches = build_batches(indices)
    labels = np.array([int(b["label"]) for b in batches])
    norm = load_norm()

    manifest = []
    for k, b in zip(indices, batches):
        manifest.append({
            "state_index": int(k),
            "label": int(b["label"]),
            "state_x": float(b["states"][0, 0]),
            "state_y": float(b["states"][0, 1]),
            "state_yaw": float(b["states"][0, 2]),
            "history_valid_steps": int(b["history_mask"][0].sum()),
            "target_norm": float(np.linalg.norm(b["target_emb"][0])),
            "current_shape": list(b["current"].shape),
        })
    import pandas as pd
    pd.DataFrame(manifest).to_csv(outdir / "shared_states_manifest.csv", index=False)

    rows = []
    for key in KEYS:
        ck = CKPTS[key]
        if not ck.exists():
            print(f"SKIP {key}: missing {ck}")
            continue
        model = build_model(key).to(args.device)
        st = torch.load(ck, map_location=args.device, weights_only=False)
        model.load_state_dict(st["model"])
        model.eval()
        for pos, (k, b) in enumerate(zip(indices, batches)):
            bb = {kk: (vv.copy() if isinstance(vv, np.ndarray) else vv)
                  for kk, vv in b.items()}
            if key in NORMALIZED:
                bb = apply_norm(bb, norm)
            mb = to_model_batch11(bb, None, args.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(mb)
            logits = logits.float()[0]
            probs = F.softmax(logits, -1)
            pred = int(probs.argmax())
            ent = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
            rows.append({
                "state_index": int(k), "label": int(b["label"]),
                "label_name": CLASSES[int(b["label"])],
                "model": key, "model_name": MODEL_NAMES[key],
                "logit_stop": float(logits[0]), "logit_forward": float(logits[1]),
                "logit_left": float(logits[2]), "logit_right": float(logits[3]),
                "p_stop": float(probs[0]), "p_forward": float(probs[1]),
                "p_left": float(probs[2]), "p_right": float(probs[3]),
                "action": pred, "action_name": CLASSES[pred],
                "confidence": float(probs[pred]), "entropy": ent,
            })
        del model
        torch.cuda.empty_cache()
        print(f"done {key}: {len(indices)} states", flush=True)

    pred = pd.DataFrame(rows)
    pred.to_csv(outdir / "expert_action_predictions.csv", index=False)

    # agreement summary
    piv = pred.pivot(index="state_index", columns="model", values="action")
    piv = piv.dropna()
    agree_all = int((piv.nunique(axis=1) == 1).sum())
    summary = {
        "n_states": int(len(piv)),
        "n_models": int(piv.shape[1]),
        "all_agree": agree_all,
        "all_agree_pct": agree_all / max(len(piv), 1),
        "any_disagreement": int(len(piv) - agree_all),
        "any_disagreement_pct": (len(piv) - agree_all) / max(len(piv), 1),
        "pairwise_action_agreement": {},
    }
    models = list(piv.columns)
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            summary["pairwise_action_agreement"][f"{a}|{b}"] = float(
                (piv[a] == piv[b]).mean())
    (outdir / "phaseB_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
