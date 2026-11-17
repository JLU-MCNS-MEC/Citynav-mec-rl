#!/usr/bin/env python3
"""Stage 14-R4: real-data smoke test for the five R2 preset MEC models.

Per model: at most 100 real training micro-batches (Large: batch 64, grad
accumulation 2), BF16 autocast, AdamW, validation macro-F1, checkpoint
save/reload.  No full training, no closed-loop, no DQN/PPO, no original
checkpoint is read or overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments/11_teacher_geometry_ablation"
                       "/11B_transformer_m_v4/scripts"))

from models import MODEL_NAMES, build_model  # noqa: E402
from dataset11b import ChunkedMemmapDataset, ValMemmapDataset  # noqa: E402
from train_11b import (build_blank_mask, collate11, evaluate11,  # noqa: E402
                       to_model_batch11)
from dataset_v2 import FrozenEncoder  # noqa: E402

KEYS = ["small", "medium", "ts", "tm", "large"]
LR = 3e-4
WD = 0.01
BLANK_FRAC = 0.05


def build_train_index(seed=42):
    root = (REPO / "experiments/11_teacher_geometry_ablation"
            "/11B_transformer_m_v4/results/data/train_seen_memmap")
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    n_train = int(meta["n_states"])
    ep_ranges = json.loads((root / "ep_ranges.json").read_text(encoding="utf-8"))
    ep_ranges = {int(k): v for k, v in ep_ranges.items()}
    assert sum(e - s for s, e in ep_ranges.values()) == n_train
    ep_list = sorted(ep_ranges)
    rng = np.random.default_rng(seed * 1000 + 0)   # Stage 11 epoch-0 order
    ep_order = rng.permutation(ep_list)
    idx = np.concatenate([np.arange(s, e, dtype=np.int64)
                          for k in ep_order for s, e in [ep_ranges[k]]])
    blank = build_blank_mask(seed, 0, ep_order, ep_ranges, BLANK_FRAC)
    return idx, blank, n_train


def run_one(key, args, idx, blank, val_ds, encoder, device):
    name = MODEL_NAMES[key]
    batch = 64 if key == "large" else 128
    accum = 2 if key == "large" else 1
    update_steps = max(1, args.steps // accum)
    model = build_model(key).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=update_steps, pct_start=0.03,
        anneal_strategy="cos", cycle_momentum=False)
    ds = ChunkedMemmapDataset("train_seen", idx, blank_mask=blank)
    loader = DataLoader(ds, batch_size=batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate11,
                        persistent_workers=False,
                        prefetch_factor=4 if args.workers else None)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    opt.zero_grad(set_to_none=True)
    hist = []
    losses, times, gnorms = [], [], []
    micro = updates = nan_inf = 0
    first_shapes = None
    t_start = time.perf_counter()
    for batch_data in loader:
        micro += 1
        t0 = time.perf_counter()
        if first_shapes is None:
            first_shapes = {k: list(v.shape) for k, v in batch_data.items()
                            if hasattr(v, "shape")}
        mb = to_model_batch11(batch_data, encoder, device)
        label = torch.from_numpy(batch_data["label"]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(mb)
            loss = F.cross_entropy(logits, label)
        loss_f = float(loss.detach())
        finite = bool(torch.isfinite(loss).item())
        if finite:
            (loss / accum).backward()
        else:
            nan_inf += 1
        update = (micro % accum == 0)
        gnorm = None
        if update:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gnorm = float(gn)
            if finite and torch.isfinite(gn):
                opt.step()
                sched.step()
                updates += 1
            else:
                nan_inf += 1
            opt.zero_grad(set_to_none=True)
        step_ms = (time.perf_counter() - t0) * 1000
        lr = float(sched.get_last_lr()[0])
        rec = {"micro_step": micro, "update": update, "loss": loss_f,
               "lr": lr, "grad_norm": gnorm, "step_ms": step_ms,
               "nan_inf": not finite}
        hist.append(rec)
        losses.append(loss_f)
        times.append(step_ms)
        if gnorm is not None and math.isfinite(gnorm):
            gnorms.append(gnorm)
        if micro >= args.steps:
            break
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    # validation macro-F1
    model.eval()
    val_batch = 64 if key == "large" else 128
    try:
        val = evaluate11(model, encoder, val_ds, device, batch_size=val_batch)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        val = evaluate11(model, encoder, val_ds, device, batch_size=16)
    # checkpoint save + reload
    out_dir = Path(args.out)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{key}.pt"
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    opt_sd = opt.state_dict()
    for st in opt_sd["state"].values():
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.cpu()
    torch.save({"model": sd, "optimizer": opt_sd,
                "scheduler": sched.state_dict(), "micro": micro,
                "updates": updates,
                "config": {"model": name, "batch": batch, "accum": accum,
                           "steps": args.steps, "lr": LR, "wd": WD}},
               ckpt_path)
    ckpt_mb = ckpt_path.stat().st_size / 1e6
    fresh = build_model(key)
    ck2 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    fresh.load_state_dict(ck2["model"])
    resume_ok = all(torch.equal(sd[k], fresh.state_dict()[k]) for k in sd)
    del fresh, ck2, sd, opt_sd
    del loader, ds
    del opt, sched
    del model
    torch.cuda.empty_cache()
    # summary
    summary = {
        "model": name, "key": key, "params": params, "batch": batch,
        "accum": accum, "micro_steps": micro, "updates": updates,
        "requested_steps": args.steps,
        "loss_first": losses[0] if losses else None,
        "loss_last10_mean": float(np.mean(losses[-10:])) if losses else None,
        "loss_min": float(np.min(losses)) if losses else None,
        "loss_max": float(np.max(losses)) if losses else None,
        "macro_f1": val.get("macro_f1"),
        "val_loss": val.get("loss"),
        "grad_norm_mean": float(np.mean(gnorms)) if gnorms else None,
        "grad_norm_max": float(np.max(gnorms)) if gnorms else None,
        "grad_norm_min": float(np.min(gnorms)) if gnorms else None,
        "step_ms_mean": float(np.mean(times)) if times else None,
        "step_ms_median": float(np.median(times)) if times else None,
        "peak_mem_gb": peak_gb,
        "nan_inf": nan_inf,
        "checkpoint": str(ckpt_path), "checkpoint_mb": ckpt_mb,
        "resume_ok": resume_ok,
        "input_shapes": first_shapes,
        "tokens": (104 if key in ("ts", "tm", "large") else "n/a"),
        "elapsed_s": round(time.perf_counter() - t_start, 2),
    }
    res_dir = out_dir / "results" / key
    res_dir.mkdir(parents=True, exist_ok=True)
    with (res_dir / "history.jsonl").open("w", encoding="utf-8") as f:
        for r in hist:
            f.write(json.dumps(r) + "\n")
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    print(f"[{name}] params={params:,} batch={batch}x{accum} "
          f"micro={micro} updates={updates} "
          f"loss {summary['loss_first']:.4f}->{summary['loss_last10_mean']:.4f} "
          f"F1={summary['macro_f1']:.4f} gnorm={summary['grad_norm_mean']:.3f} "
          f"step={summary['step_ms_mean']:.1f}ms peak={peak_gb:.2f}GB "
          f"nan={nan_inf} resume={resume_ok} ckpt={ckpt_mb:.0f}MB",
          flush=True)
    return summary


def write_report(args, summaries):
    out = Path(args.out)
    lines = ["# Stage 14-R4 真实数据冒烟测试报告", "",
             "日期：2026-11-17", "",
             "每个模型最多 100 个真实训练 micro-batch；Large 为 batch 64 × 累积 2。",
             "**100 steps 不代表收敛**；本报告只验证数据/前向/反向/优化器/验证/",
             "Checkpoint 链路。", "",
             "| 模型 | 参数 | Batch×累积 | 更新步 | Loss first→last10 | Val Macro-F1 | Grad norm mean/max | Step ms | Peak GB | NaN/Inf | Ckpt MB | Resume |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for s in summaries:
        lines.append(
            f"| {s['model']} | {s['params']:,} | {s['batch']}×{s['accum']} | "
            f"{s['updates']} | {s['loss_first']:.4f}→{s['loss_last10_mean']:.4f} | "
            f"{s['macro_f1']:.4f} | {s['grad_norm_mean']:.3f}/{s['grad_norm_max']:.3f} | "
            f"{s['step_ms_mean']:.1f} | {s['peak_mem_gb']:.2f} | "
            f"{s['nan_inf']} | {s['checkpoint_mb']:.0f} | "
            f"{'OK' if s['resume_ok'] else 'FAIL'} |")
    lines += ["", "## 判定", ""]
    ok_all = True
    for s in summaries:
        checks = [
            s["nan_inf"] == 0,
            s["macro_f1"] is not None and s["macro_f1"] > 0,
            s["grad_norm_mean"] is not None and math.isfinite(s["grad_norm_mean"]),
            s["resume_ok"],
            s["peak_mem_gb"] < 32.0,
            s["updates"] >= max(1, s["requested_steps"] // s["accum"]),
        ]
        passed = all(checks)
        ok_all &= passed
        lines.append(f"- {s['model']}: {'PASS' if passed else 'FAIL'} "
                     f"(NaN/Inf={s['nan_inf']}, F1={s['macro_f1']:.4f}, "
                     f"resume={s['resume_ok']}, peak={s['peak_mem_gb']:.2f}GB)")
    lines += ["", f"总体：{'全部通过' if ok_all else '存在失败项'}。" , "",
              "产物：`results/<model>/{history.jsonl,summary.json}`、",
              "`checkpoints/<model>.pt`。未训练完整 epoch，未跑闭环/DQN/PPO。" , ""]
    (out / "SMOKE_TEST_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out/'SMOKE_TEST_REPORT.md'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["all"],
                    choices=["all"] + KEYS)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(HERE / "smoke_test"))
    a = ap.parse_args()
    keys = KEYS if a.models == ["all"] else a.models
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    idx, blank, n_train = build_train_index(a.seed)
    print(f"train states {n_train} | iterating {a.steps} micro-batches per model",
          flush=True)
    val_ds = ValMemmapDataset("val_seen")
    encoder = FrozenEncoder(a.device)
    summaries = []
    for key in keys:
        summaries.append(run_one(key, a, idx, blank, val_ds, encoder,
                                 a.device))
    write_report(a, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
