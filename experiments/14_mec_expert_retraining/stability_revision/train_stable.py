#!/usr/bin/env python3
"""Stage 14-R2: unified stable training for Transformer-S / Transformer-M / Large.

Shared protocol (identical for all three):
  AdamW, max_lr=5e-5, wd=0.01, OneCycleLR (warmup=10%, cosine), clip=1.0,
  BF16 autocast, <=20 epochs, patience=10, Dev Macro-F1 best selection.

Shared normalization (Train-only statistics, see normalization_statistics.json):
  belief (10 dims), match (5 dims), history_pose (3 dims) are standardized.
  current / history_vis / target_emb come from the frozen encoder and are
  untouched; landmarks/relation keep the V5FIX geometric normalization;
  states stay in world coordinates (relative geometry only).

Shared initialization:
  token/position/type embeddings -> Normal(0, 0.02); Linear/LayerNorm keep the
  PyTorch defaults; Pre-LN structure unchanged.

Stability protection:
  * non-finite or pre-clip grad norm > 1e4  -> update skipped (never clipped
    into a "valid" update);
  * 10 consecutive skipped updates -> abort with rc=2;
  * per-step grad_trace.jsonl (pre/post clip norm, LR, weight norm, skip flag);
  * sampled attention/residual diagnostics.

Outputs go to stability_revision/runs/<model>/seed<seed>/ and NEVER touch the
original checkpoints/ directory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent          # stability_revision/
EXP = HERE.parent                               # 14_mec_expert_retraining/
REPO = EXP.parents[1]
SCRIPTS = REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4/scripts"
for p in [EXP, SCRIPTS,
          REPO / "experiments/10_goal_query_mec_training/10D_transformer_m_training/scripts",
          REPO / "experiments/05_transformer_m_training/scripts",
          REPO / "experiments/04_visual_goal_global_map_pipeline/scripts"]:
    sys.path.insert(0, str(p))

from models import MODEL_NAMES, build_model                       # noqa: E402
from dataset11b import ChunkedMemmapDataset                       # noqa: E402
from train_11b import collate11, to_model_batch11, build_blank_mask  # noqa: E402
from train_10d import metrics, rng_state, restore_rng, set_seed   # noqa: E402

DATA = REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4/results/data/train_seen_memmap"
CLASSES = ["Stop", "Forward", "TurnLeft", "TurnRight"]
KEYS = ["ts", "tm", "large"]
TRAIN_BATCH = {"ts": (128, 1), "tm": (128, 1), "large": (64, 2)}

# ---- unified protocol -------------------------------------------------------
DEFAULTS = dict(seed=42, epochs=20, lr=5e-5, weight_decay=0.01, warmup=0.10,
                patience=10, blank_frac=0.05, dev_frac=0.02, workers=8,
                clip=1.0, skip_gn=1e4, max_consec_skips=10,
                diag_every=500, weight_norm_every=1)


def load_ep_ranges() -> dict[int, tuple[int, int]]:
    ep = json.loads((DATA / "ep_ranges.json").read_text(encoding="utf-8"))
    return {int(k): (int(v[0]), int(v[1])) for k, v in ep.items()}


def make_dev_split(ep, seed, dev_frac):
    eps = np.asarray(sorted(ep), dtype=np.int64)
    rng = np.random.default_rng(seed * 7777 + 14)
    perm = rng.permutation(eps)
    n_dev = max(1, int(round(len(eps) * dev_frac)))
    return sorted(int(x) for x in perm[:n_dev])


def split_states(ep, keys):
    return np.concatenate([np.arange(ep[k][0], ep[k][1], dtype=np.int64) for k in keys])


# ---- shared normalization ---------------------------------------------------
def load_norm(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))["standardized"]
    return {k: (np.asarray(d[k]["mean"], np.float32),
                np.asarray(d[k]["std"], np.float32))
            for k in ("belief", "match", "history_pose")}


def normalize_np(batch: dict, norm: dict) -> dict:
    for key, (mean, std) in norm.items():
        if key in batch:
            batch[key] = ((batch[key] - mean) / std).astype(np.float32)
    return batch


# ---- shared initialization --------------------------------------------------
TOKEN_PARAM_NAMES = {
    "cls_token", "cls", "current_pos", "vis_pos", "goal_pos",
    "hist_time", "current_type", "goal_type", "goal_pos",
}


def uniform_init(model: nn.Module, std: float = 0.02) -> None:
    """Token/position/type parameters -> Normal(0, std).

    Linear / LayerNorm keep the PyTorch defaults (xavier_uniform / ones+zeros).
    Only parameters named like explicit token or position embeddings are reset,
    plus every nn.Embedding weight (type_emb, hist_action, ...).
    """
    reset = []
    for name, p in model.named_parameters():
        leaf = name.split(".")[-2] if name.endswith(".weight") else name.split(".")[-1]
        if leaf in TOKEN_PARAM_NAMES or name in TOKEN_PARAM_NAMES:
            nn.init.normal_(p, mean=0.0, std=std)
            reset.append(name)
    for mod_name, mod in model.named_modules():
        if isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, mean=0.0, std=std)
            reset.append(mod_name + ".weight")
    print(f"[init] Normal(0,{std}) applied to {len(reset)} tensors", flush=True)
    for n in sorted(reset):
        print(f"  init {n}", flush=True)


# ---- diagnostics ------------------------------------------------------------
SLICES = {"cls": (0, 1), "current": (1, 50), "target": (50, 51),
          "history": (51, 67), "state": (67, 68), "landmark": (68, 76),
          "relation": (76, 104)}


def encoder_module(model):
    return model.fusion_encoder if hasattr(model, "fusion_encoder") else model.encoder


def enable_gradient_checkpointing(model) -> int:
    """Trade ~20-30% step time for a large cut in activation memory.

    Does not change parameters, architecture, batch composition, or LR; only
    the backward pass recomputes each encoder layer's forward.
    """
    import torch.utils.checkpoint as cp
    enc = encoder_module(model)
    for layer in enc.layers:
        orig = layer.forward

        def wrapped(*args, __orig=orig, **kwargs):
            return cp.checkpoint(__orig, *args, use_reentrant=False, **kwargs)

        layer.forward = wrapped
    return len(enc.layers)


def token_scale_report(model, batch_t, device) -> dict:
    """Token statistics at the input of the Transformer encoder."""
    enc = encoder_module(model)
    captured = {}

    def pre_hook(mod, args, kwargs):
        captured["seq"] = args[0].detach()

    h = enc.register_forward_pre_hook(pre_hook, with_kwargs=True)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(batch_t)
    h.remove()
    seq = captured["seq"].float()
    out = {"n_tokens": int(seq.shape[1])}
    for name, (a, b) in SLICES.items():
        t = seq[:, a:b]
        norm = t.norm(dim=-1)
        out[name] = {
            "mean": float(t.mean()), "std": float(t.std()),
            "mean_l2": float(norm.mean()), "max_l2": float(norm.max()),
            "nan": int(torch.isnan(t).sum()), "inf": int(torch.isinf(t).sum()),
        }
    model.train()
    return out


def attention_report(model, batch_t, device) -> dict:
    """Per-layer Q/K/V std, QK-score stats, attention entropy/max, residual std."""
    enc = encoder_module(model)
    caps = []
    handles = []

    def make_pre(name):
        def hook(mod, args, kwargs):
            x = args[0] if args else kwargs.get("query")
            kpm = kwargs.get("key_padding_mask", None)
            W = mod.in_proj_weight
            bias = mod.in_proj_bias
            nh = mod.num_heads
            b_, T, d = x.shape
            dh = d // nh
            with torch.no_grad():
                qkv = F.linear(x.float(), W.float(), bias.float()).chunk(3, -1)
                q, k, v = [t.view(b_, T, nh, dh).transpose(1, 2) for t in qkv]
                scores = q @ k.transpose(-2, -1) / math.sqrt(dh)
                if kpm is not None:
                    # autocast canonicalizes the bool padding mask to a float
                    # tensor (True -> 1.0); normalize back to bool.
                    kpm_bool = kpm if kpm.dtype == torch.bool else (kpm != 0)
                    scores = scores.masked_fill(kpm_bool[:, None, None, :],
                                                float("-inf"))
                attn = torch.softmax(scores, -1)
                ent = -(attn * torch.log(attn.clamp_min(1e-12))).sum(-1).mean()
                caps.append({
                    "layer": name,
                    "q_std": float(q.std()), "k_std": float(k.std()),
                    "v_std": float(v.std()),
                    "score_std": float(scores[torch.isfinite(scores)].std()),
                    "score_absmax": float(scores[torch.isfinite(scores)].abs().max()),
                    "attn_entropy": float(ent),
                    "attn_max": float(attn.max()),
                })
        return hook

    def make_post(name):
        def hook(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            caps.append({"layer": name + ".out", "residual_std": float(x.float().std()),
                         "residual_absmax": float(x.float().abs().max())})
        return hook

    for i, layer in enumerate(enc.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(
            make_pre(f"L{i}"), with_kwargs=True))
        handles.append(layer.register_forward_hook(make_post(f"L{i}")))
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(batch_t)
    for h in handles:
        h.remove()
    model.train()
    return {"layers": caps}


def weight_norm(model) -> float:
    with torch.no_grad():
        return float(torch.sqrt(sum(p.detach().float().norm(2).pow(2)
                                    for p in model.parameters())))


# ---- data -------------------------------------------------------------------
def make_train_loader(idx, blank_mask, batch, workers):
    ds = ChunkedMemmapDataset("train_seen", idx, blank_mask=blank_mask)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                      collate_fn=collate11, persistent_workers=workers > 0,
                      prefetch_factor=4 if workers else None)


@torch.no_grad()
def evaluate_dev(model, dev_idx, norm, device, batch=128) -> dict:
    model.eval()
    ds = ChunkedMemmapDataset("train_seen", dev_idx, blank_mask=None)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4,
                        collate_fn=collate11)
    preds, labels, losses = [], [], []
    for b in loader:
        b = normalize_np(b, norm)
        label = torch.from_numpy(b["label"]).to(device)
        mb = to_model_batch11(b, None, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(mb)
            loss = F.cross_entropy(logits, label)
        losses.append(float(loss) * len(label))
        preds.extend(logits.argmax(-1).cpu().numpy().tolist())
        labels.extend(b["label"].tolist())
    return metrics(np.asarray(preds), np.asarray(labels),
                   float(np.sum(losses) / max(len(labels), 1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=KEYS)
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    ap.add_argument("--warmup", type=float, default=DEFAULTS["warmup"])
    ap.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    ap.add_argument("--blank-frac", type=float, default=DEFAULTS["blank_frac"])
    ap.add_argument("--dev-frac", type=float, default=DEFAULTS["dev_frac"])
    ap.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    ap.add_argument("--clip", type=float, default=DEFAULTS["clip"])
    ap.add_argument("--skip-gn", type=float, default=DEFAULTS["skip_gn"])
    ap.add_argument("--max-consec-skips", type=int, default=DEFAULTS["max_consec_skips"])
    ap.add_argument("--diag-every", type=int, default=DEFAULTS["diag_every"])
    ap.add_argument("--weight-norm-every", type=int, default=DEFAULTS["weight_norm_every"])
    ap.add_argument("--norm-file", default=str(HERE / "normalization_statistics.json"))
    ap.add_argument("--resume", default="",
                    help="path to a last.pt to continue training from "
                         "(restores model, optimizer, RNG, best, history)")
    ap.add_argument("--scheduler", choices=["onecycle", "cosine_cont"],
                    default="onecycle",
                    help="onecycle = original Stage-14 schedule; cosine_cont = "
                         "low-LR continuation without warmup (Stage 14-R3)")
    ap.add_argument("--lr-end", type=float, default=5e-7,
                    help="cosine_cont final LR (eta_min)")
    ap.add_argument("--min-epochs", type=int, default=0,
                    help="do not early-stop before this epoch (0 = patience only)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="per-layer gradient checkpointing (activation-memory "
                         "trade; does not alter architecture/params/batch/LR)")
    ap.add_argument("--out-dir", default=str(HERE / "runs"))
    ap.add_argument("--device", default="cuda")
    # engineering-only smoke controls (0 = full)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--dev-limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()

    if not os.environ.get("CITYNAV_DATA"):
        raise SystemExit("CITYNAV_DATA is not set")
    key, name = args.model, MODEL_NAMES[args.model]
    batch, accum = TRAIN_BATCH[key]
    set_seed(args.seed)
    device = args.device

    run_dir = Path(args.out_dir) / key / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    best_path, last_path = ckpt_dir / "best.pt", ckpt_dir / "last.pt"

    norm = load_norm(Path(args.norm_file))
    ep = load_ep_ranges()
    dev_eps = make_dev_split(ep, args.seed, args.dev_frac)
    dev_set = set(dev_eps)
    train_eps = [k for k in sorted(ep) if k not in dev_set]
    dev_idx = split_states(ep, dev_eps)
    if args.dev_limit:
        dev_idx = dev_idx[: args.dev_limit]
    n_train = int(sum(ep[k][1] - ep[k][0] for k in train_eps))
    updates_per_epoch = math.ceil(n_train / (batch * accum))

    resume_info = None
    remaining_epochs = args.epochs
    config = {
        "stage": "14-R2-stability", "model": name, "key": key,
        "seed": args.seed, "lr": args.lr, "weight_decay": args.weight_decay,
        "optimizer": "AdamW",
        "scheduler": ("OneCycleLR" if args.scheduler == "onecycle"
                      else "CosineAnnealingLR-continuation"),
        "lr_end": args.lr_end, "min_epochs": args.min_epochs,
        "warmup_ratio": args.warmup, "anneal": "cos",
        "clip": args.clip, "skip_gn": args.skip_gn,
        "precision": "bf16-autocast", "epochs": args.epochs,
        "patience": args.patience, "batch": batch, "accum": accum,
        "effective_batch": batch * accum,
        "steps_per_epoch": math.ceil(n_train / batch),
        "updates_per_epoch": updates_per_epoch,
        "n_train_states": n_train, "n_train_episodes": len(train_eps),
        "n_dev_states": len(dev_idx), "n_dev_episodes": len(dev_eps),
        "normalizer": str(args.norm_file),
        "resume": resume_info,
        "remaining_epochs": remaining_epochs,
        "grad_checkpoint": bool(args.grad_checkpoint),
        "init": "token/pos/type Normal(0,0.02); Linear/LayerNorm defaults",
        "loss": "cross_entropy (4-class, no class weight, no label smoothing)",
        "git_sha": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    model = build_model(key).to(device)
    uniform_init(model)
    if args.grad_checkpoint:
        n_ck = enable_gradient_checkpointing(model)
        print(f"[{name}] gradient checkpointing enabled on {n_ck} layers",
              flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] params={n_params:,} batch={batch}x{accum} "
          f"updates/epoch={updates_per_epoch:,} lr={args.lr} warmup={args.warmup}",
          flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    start_epoch, best, best_epoch, n_best_epochs, global_step = 0, -1.0, -1, 0, 0
    hist: list[dict] = []
    resume_info = None
    if args.resume:
        rp = Path(args.resume)
        if not rp.exists():
            raise SystemExit(f"--resume path does not exist: {rp}")
        ck = torch.load(rp, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        best = float(ck.get("best", -1.0))
        global_step = int(ck.get("global_step", 0))
        start_epoch = int(ck["epoch"]) + 1
        if "rng" in ck:
            restore_rng(ck["rng"])
        hp = run_dir / "training_history.jsonl"
        if hp.exists():
            for line in hp.read_text(encoding="utf-8").splitlines():
                try:
                    hist.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if hist:
            f1s = [h["dev"]["macro_f1"] for h in hist]
            best_epoch = int(np.argmax(f1s))
            best = max(best, float(f1s[best_epoch]))
            n_best_epochs = len(f1s) - 1 - best_epoch
        resume_info = {"path": str(rp), "ckpt_epoch": int(ck["epoch"]),
                       "ckpt_global_step": global_step}
        print(f"[{name}] resumed from {rp} (epoch {ck['epoch']} -> "
              f"start_epoch {start_epoch}, best={best:.4f}, "
              f"global_step={global_step})", flush=True)

    remaining_epochs = max(1, args.epochs - start_epoch)
    total_updates = max(1, remaining_epochs * updates_per_epoch)
    if args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_updates,
            pct_start=args.warmup, anneal_strategy="cos", cycle_momentum=False)
        print(f"[{name}] schedule: OneCycle {remaining_epochs} epochs x "
              f"{updates_per_epoch} updates, warmup "
              f"{int(args.warmup*total_updates)}", flush=True)
    else:
        # Stage 14-R3: continuation at a low LR, cosine decay 5e-6 -> 5e-7,
        # no warmup, and the optimizer state (AdamW moments) is resumed as-is.
        for g in optimizer.param_groups:
            g["lr"] = args.lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_updates, eta_min=args.lr_end)
        print(f"[{name}] schedule: cosine_cont {args.lr:.1e} -> "
              f"{args.lr_end:.1e} over {total_updates} updates, no warmup",
              flush=True)
    config["resume"] = resume_info
    config["remaining_epochs"] = remaining_epochs
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    trace = (run_dir / "grad_trace.jsonl").open("a", encoding="utf-8")
    attn_diag = (run_dir / "attention_diagnostics.jsonl").open("a", encoding="utf-8")
    tok_diag_path = run_dir / "token_scale_diagnostics.json"

    consec_skips = 0
    total_skips = 0
    nan_losses = 0
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()
        rng = np.random.default_rng(args.seed * 1000 + epoch)
        order = rng.permutation(np.asarray(train_eps, dtype=np.int64))
        idx = np.concatenate([np.arange(ep[k][0], ep[k][1], dtype=np.int64)
                              for k in order])
        blank = build_blank_mask(args.seed, epoch, order, ep, args.blank_frac)
        if args.train_limit:
            idx, blank = idx[: args.train_limit], blank[: args.train_limit]
        loader = make_train_loader(idx, blank, batch, args.workers)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, count, correct, updates, micro = 0.0, 0, 0, 0, 0
        n_batches = len(loader)

        for bi, b in enumerate(loader):
            micro += 1
            b = normalize_np(b, norm)
            label = torch.from_numpy(b["label"]).to(device)
            mb = to_model_batch11(b, None, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(mb)
                loss = F.cross_entropy(logits, label)
            if not bool(torch.isfinite(loss).item()):
                nan_losses += 1
                optimizer.zero_grad(set_to_none=True)
                trace.write(json.dumps({"step": global_step, "epoch": epoch,
                                        "micro": micro, "nan_loss": True}) + "\n")
                continue
            (loss / accum).backward()
            total += float(loss.detach()) * len(label)
            correct += int((logits.detach().argmax(-1) == label).sum())
            count += len(label)

            if micro % accum == 0:
                w_before = weight_norm(model) if (micro // accum) % args.weight_norm_every == 0 else None
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip))
                bad = (not math.isfinite(gn)) or (gn > args.skip_gn)
                if bad:
                    total_skips += 1
                    consec_skips += 1
                    optimizer.zero_grad(set_to_none=True)
                    trace.write(json.dumps({
                        "step": global_step, "epoch": epoch, "micro": micro,
                        "lr": float(scheduler.get_last_lr()[0]),
                        "grad_norm_preclip": gn if math.isfinite(gn) else "inf",
                        "skipped": True, "consecutive_skips": consec_skips,
                    }) + "\n", )
                    if consec_skips >= args.max_consec_skips:
                        trace.flush()
                        msg = (f"ABORT: {consec_skips} consecutive pathological "
                               f"gradient updates (last pre-clip norm {gn})")
                        print(msg, flush=True)
                        (run_dir / "ABORT.json").write_text(json.dumps(
                            {"reason": msg, "epoch": epoch, "mirco": micro,
                             "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2))
                        trace.close(); attn_diag.close()
                        return 2
                    if args.max_steps and micro >= args.max_steps:
                        break
                    continue
                consec_skips = 0
                grad_norms = [p.grad for p in model.parameters() if p.grad is not None]
                nonfinite = sum(int((~torch.isfinite(g)).sum()) for g in grad_norms)
                optimizer.step()
                scheduler.step()
                updates += 1
                global_step += 1
                w_after = weight_norm(model) if (micro // accum) % args.weight_norm_every == 0 else None
                trace.write(json.dumps({
                    "step": global_step, "epoch": epoch, "micro": micro,
                    "lr": float(scheduler.get_last_lr()[0]),
                    "grad_norm_preclip": gn,
                    "grad_norm_postclip": float(min(gn, args.clip)),
                    "weight_norm_before": w_before,
                    "weight_norm_after": w_after,
                    "skipped": False,
                    "nonfinite_grad_params": nonfinite,
                }) + "\n")
                optimizer.zero_grad(set_to_none=True)

                if args.diag_every and global_step % args.diag_every == 0:
                    try:
                        rep = attention_report(model, mb, device)
                        rep.update({"step": global_step, "epoch": epoch,
                                    "model": key})
                        attn_diag.write(json.dumps(rep) + "\n")
                        attn_diag.flush()
                        L0 = next((x for x in rep["layers"]
                                   if x.get("layer") == "L0"), {})
                        print(f"  diag step {global_step}: q_std={L0.get('q_std'):.3f} "
                              f"score_absmax={L0.get('score_absmax'):.3e} "
                              f"attn_max={L0.get('attn_max'):.4f} "
                              f"entropy={L0.get('attn_entropy'):.3f}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        attn_diag.write(json.dumps({"step": global_step,
                                                    "error": str(e)}) + "\n")

            if updates and updates % 200 == 0 and micro % accum == 0:
                el = time.time() - t_epoch
                print(f"  epoch {epoch} update {updates}/{updates_per_epoch} "
                      f"loss={total / max(count, 1):.4f} "
                      f"acc={correct / max(count, 1):.4f} {el:.0f}s "
                      f"lr={scheduler.get_last_lr()[0]:.2e} skips={total_skips}",
                      flush=True)
            if args.max_steps and micro >= args.max_steps:
                break
        trace.flush()

        train_loss = total / max(count, 1)
        train_acc = correct / max(count, 1)
        dev = evaluate_dev(model, dev_idx, norm, device)
        rec = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
               "micro_steps": micro, "optimizer_steps": updates,
               "global_step": global_step, "dev": dev,
               "lr": float(scheduler.get_last_lr()[0]),
               "seconds": round(time.time() - t_epoch, 1),
               "skipped_updates": total_skips, "nan_losses": nan_losses,
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
               "gpu": gpu_snapshot()}
        hist.append(rec)
        with (run_dir / "training_history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        (run_dir / "training_history.json").write_text(json.dumps(hist, indent=1) + "\n")
        tok = token_scale_report(model, mb, device)
        (run_dir / f"token_scale_diagnostics_epoch{epoch}.json").write_text(
            json.dumps(tok, indent=1) + "\n")
        if epoch == 0:
            tok_diag_path.write_text(json.dumps(
                {"note": "token scale at encoder input; epoch 0 end", **tok},
                indent=1) + "\n")

        improved = dev["macro_f1"] > best
        if improved:
            best, best_epoch, n_best_epochs = float(dev["macro_f1"]), epoch, 0
            save_atomic({"model": model.state_dict(), "config": config,
                         "epoch": epoch, "dev": dev, "macro_f1": best}, best_path)
        else:
            n_best_epochs += 1
        save_atomic({"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "epoch": epoch,
                     "epoch_done": True, "config": config, "best": best,
                     "global_step": global_step, "rng": rng_state()}, last_path)
        (run_dir / "train_progress.json").write_text(json.dumps({
            "model": name, "key": key, "epoch": epoch, "epochs": args.epochs,
            "optimizer_step": global_step, "train_loss": train_loss,
            "dev_macro_f1": dev["macro_f1"], "best": best,
            "best_epoch": best_epoch, "lr": rec["lr"],
            "skipped_updates": total_skips, "patience_left": args.patience - n_best_epochs,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2) + "\n")
        print(json.dumps({"epoch": epoch, "loss": round(train_loss, 4),
                          "acc": round(train_acc, 4),
                          "dev_macro_f1": round(dev["macro_f1"], 4),
                          "dev_acc": round(dev["accuracy"], 4),
                          "pred": [dev["per_class"][c]["predicted"] for c in CLASSES],
                          "lr": round(rec["lr"], 8), "best": round(best, 4),
                          "skips": total_skips, "sec": rec["seconds"]}), flush=True)
        if n_best_epochs >= args.patience:
            if args.min_epochs and epoch + 1 < args.min_epochs:
                print(f"[{name}] patience reached at epoch {epoch} but "
                      f"min_epochs={args.min_epochs}; continuing", flush=True)
            else:
                print(f"early stop (patience {args.patience})", flush=True)
                break

    trace.close()
    attn_diag.close()
    (run_dir / "DONE.json").write_text(json.dumps({
        "model": name, "epochs_run": len(hist), "best": best,
        "best_epoch": best_epoch, "global_step": global_step,
        "skipped_updates": total_skips, "nan_losses": nan_losses,
        "n_params": n_params, "elapsed_sec": round(time.time() - t_start, 1),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2) + "\n")
    return 0


def save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def gpu_snapshot() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
        u, m, t = (x.strip() for x in out.splitlines()[0].split(","))
        return {"gpu_util_pct": int(u), "mem_mib": int(m), "temp_c": int(t)}
    except Exception:  # noqa: BLE001
        return {}


if __name__ == "__main__":
    raise SystemExit(main())
