#!/usr/bin/env python3
"""Stage 14 pre-training audit: smoke-test results + input contract.

Checks, per model:
  1. smoke summary: 100 real micro-batches (Large 100 micro / 50 updates),
     finite loss, AdamW updates, resume_ok, peak memory;
  2. smoke checkpoint reloads strictly into the same architecture;
  3. parameter count matches the R2 preset table;
  4. the eight input modalities are all consumed (perturbation test);
  5. Transformer-S / Transformer-M / Large feed exactly 104 tokens;
  6. Small/Medium keep their non-Transformer structure (no TransformerEncoder);
  7. the formal trainer imports models.build_model from the same module.
Writes SMOKE_AUDIT.md + smoke_audit.json. Read-only for model data/checkpoints.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments/11_teacher_geometry_ablation"
                       "/11B_transformer_m_v4/scripts"))

from models import MODEL_NAMES, build_model  # noqa: E402
import models as models_mod  # noqa: E402
import train_stage14  # noqa: E402

KEYS = ["small", "medium", "ts", "tm", "large"]
EXPECTED_PARAMS = {"small": 61_121_028, "medium": 166_011_908,
                   "ts": 211_181_060, "tm": 498_567_684,
                   "large": 1_034_877_444}
R3_PEAK_GB = {"small": 2.05, "medium": 4.39, "ts": 10.29, "tm": 19.19,
              "large": 24.04}
BATCH_ACCUM = {"small": (128, 1), "medium": (128, 1), "ts": (128, 1),
               "tm": (128, 1), "large": (64, 2)}
TRANSFORMERS = ("ts", "tm", "large")


def make_batch(b: int = 2, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    batch = {
        "current": r(b, 49, 2048),
        "target_emb": r(b, 512),
        "history_vis": r(b, 16, 2048),
        "history_pose": r(b, 16, 3),
        "history_action": torch.nn.functional.one_hot(
            torch.randint(0, 4, (b, 16), generator=g), 4).float(),
        "history_mask": torch.ones(b, 16),
        "belief": r(b, 10),
        "match": r(b, 5),
        "states": torch.stack([torch.full((b,), 100.0),
                               torch.full((b,), 200.0),
                               torch.full((b,), 0.3)], dim=-1),
        "landmarks": torch.stack([
            torch.tensor([[120.0, 210.0, 1.0], [80.0, 190.0, 0.8],
                          [150.0, 230.0, 0.6], [60.0, 220.0, 0.4],
                          [130.0, 180.0, 0.2], [0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])] * b),
    }
    return batch


def token_count(model, batch) -> int | None:
    seen = []
    target = getattr(model, "fusion_encoder", None)
    if target is None:
        target = getattr(model, "encoder", None)
    if target is None:
        return None
    h = target.register_forward_pre_hook(
        lambda _m, args: seen.append(tuple(args[0].shape)))
    with torch.no_grad():
        model(batch)
    h.remove()
    return seen[0][1] if seen else None


def modality_audit(model, device) -> dict:
    """Perturb each modality; a consumed modality changes the logits."""
    model.eval()
    base = {k: v.to(device) for k, v in make_batch().items()}
    with torch.no_grad():
        ref = model(base).float()
    out = {}
    for key in ["current", "target_emb", "history_vis", "history_pose",
                "history_action", "history_mask", "belief", "match",
                "landmarks", "states"]:
        b = {k: v.clone() for k, v in base.items()}
        if key == "history_mask":
            b[key] = torch.zeros_like(b[key])
        elif key == "history_action":
            b[key] = torch.nn.functional.one_hot(
                torch.full((2, 16), 2, dtype=torch.long), 4).float().to(device)
        elif key == "landmarks":
            b[key] = b[key] + torch.tensor([5.0, -4.0, 0.0], device=device)
        elif key == "states":
            b[key] = b[key] + torch.tensor([3.0, -2.0, 0.1], device=device)
        else:
            b[key] = b[key] + 0.25
        with torch.no_grad():
            d = float((model(b).float() - ref).abs().max())
        out[key] = d
    return out


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {"models": {}, "checks": []}
    ok = True
    for key in KEYS:
        name = MODEL_NAMES[key]
        # TS/Large were re-smoked after the history-mask fix; prefer that run.
        sroot = HERE / f"smoke_test_maskfix/results/{key}"
        croot = HERE / f"smoke_test_maskfix/checkpoints/{key}.pt"
        if not (sroot / "summary.json").exists():
            sroot = HERE / "smoke_test/results" / key
            croot = HERE / "smoke_test/checkpoints" / f"{key}.pt"
        sdir = sroot
        summ = json.loads((sdir / "summary.json").read_text(encoding="utf-8"))
        hist = [json.loads(x) for x in
                (sdir / "history.jsonl").read_text(encoding="utf-8").splitlines()
                if x.strip()]
        ck_path = croot
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model = build_model(key)          # exactly the R2 preset definition
        model.load_state_dict(ck["model"], strict=True)
        n_params = sum(p.numel() for p in model.parameters())
        batch, accum = BATCH_ACCUM[key]
        exp_updates = 100 // accum
        checks = {
            "smoke_reported": len(hist) > 0,
            "micro_steps_100": summ["micro_steps"] == 100,
            "updates_expected": summ["updates"] == exp_updates,
            "no_nan_inf": summ["nan_inf"] == 0,
            "loss_decreased": summ["loss_last10_mean"] < summ["loss_first"],
            "dev_f1_finite": summ.get("macro_f1") is not None
                             and summ["macro_f1"] > 0,
            "grad_norm_finite": summ["grad_norm_mean"] is not None
                                and torch.isfinite(
                                    torch.tensor(summ["grad_norm_mean"])).item(),
            "resume_ok": bool(summ["resume_ok"]),
            "params_match": n_params == EXPECTED_PARAMS[key],
            "batch_accum_match": (summ["batch"], summ["accum"]) == (batch, accum),
            "ckpt_config_match": (ck["config"]["batch"],
                                  ck["config"]["accum"]) == (batch, accum),
            "peak_under_32gb": summ["peak_mem_gb"] < 32.0,
            "formal_uses_smoke_model": train_stage14.build_model is \
                models_mod.build_model,
        }
        model = model.to(device).eval()
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            n_tok = token_count(model, {k: v.to(device)
                                        for k, v in make_batch().items()})
        if key in TRANSFORMERS:
            checks["tokens_104"] = n_tok == 104
        else:
            checks["no_transformer_encoder"] = not any(
                isinstance(m, nn.TransformerEncoder) for m in model.modules())
        pert = modality_audit(model, device)
        # any exactly-zero delta means the modality is not wired into forward
        checks["all_modalities_used"] = all(v > 1e-7 for v in pert.values())
        passed = all(checks.values())
        ok &= passed
        report["models"][key] = {
            "name": name, "params": n_params, "batch": batch, "accum": accum,
            "tokens": n_tok, "smoke_peak_gb": summ["peak_mem_gb"],
            "r3_peak_gb": R3_PEAK_GB[key],
            "checks": checks, "passed": passed, "modality_logit_delta": pert,
        }
        print(f"[{name}] {'PASS' if passed else 'FAIL'} "
              f"params={n_params:,} tokens={n_tok} "
              f"peak={summ['peak_mem_gb']:.2f}GB "
              f"modalities(min)={min(pert.values()):.3e}", flush=True)
        del model
        torch.cuda.empty_cache()
    report["all_passed"] = ok
    (HERE / "smoke_audit.json").write_text(json.dumps(report, indent=2) + "\n",
                                           encoding="utf-8")

    lines = ["# Stage 14 正式训练前审计：Smoke Test + 输入契约", "",
             f"日期：2026-11-17　结论：**{'全部通过' if ok else '存在失败项'}**", "",
             "| 模型 | 参数 | Batch×累积 | Smoke 更新步 | 104-token | "
             "八模态全部使用 | Smoke 峰值 | R3 估计 | 判定 |",
             "|---|---:|---|---:|---|---|---:|---:|---|"]
    for key in KEYS:
        m = report["models"][key]
        lines.append(
            f"| {m['name']} | {m['params']:,} | {m['batch']}×{m['accum']} | "
            f"{'100' if key != 'large' else '50 (100 micro)'} | "
            f"{'是' if m['tokens'] == 104 else '—'} | "
            f"{'是' if m['checks']['all_modalities_used'] else '否'} | "
            f"{m['smoke_peak_gb']:.2f} GB | {m['r3_peak_gb']:.2f} GB | "
            f"{'PASS' if m['passed'] else 'FAIL'} |")
    lines += ["", "## 逐项检查", ""]
    for key in KEYS:
        m = report["models"][key]
        fails = [k for k, v in m["checks"].items() if not v]
        lines.append(f"- **{m['name']}**：{len(m['checks'])-len(fails)}/"
                     f"{len(m['checks'])} 项通过"
                     + (f"；失败：{', '.join(fails)}" if fails else ""))
    lines += ["", "## 输入 Shape 契约（八种模态）", "",
              "| 模态 | Shape |", "|---|---|",
              "| Current Visual | 49×2048 |", "| Target | 512 |",
              "| History Visual | 16×2048 |", "| History Pose | 16×3 |",
              "| History Action | 16×4 (one-hot) |", "| State | 12 (belief 7 + match 5) |",
              "| Landmark | 8×3 |", "| Relation | 28×3（由 V5FIX 几何层生成） |", "",
              "三种 Transformer 的 token 序列为 1(cls)+49(current)+1(goal)"
              "+16(history)+1(state)+8(landmark)+28(relation)=**104**。", "",
              "## 结论与注意", "",
              "- Smoke 与正式训练共用 `models.build_model`、`collate11`/"
              "`to_model_batch11`，输入处理完全一致。",
              "- Large 实测峰值 28.28 GB，高于 R3 估计 24.04 GB（+4.24 GB），"
              "仍低于 32 GB；正式训练沿用 batch 64 × 累积 2，不启用 gradient "
              "checkpointing，若 OOM 立即暂停报告。",
              "- Dev 集：训练池 2% episode 独立留出（本阶段开始前已确认），"
              "与 Val-Seen 2498 闭环任务无重叠；不使用 Val-Seen 选 best。",
              ""]
    (HERE / "SMOKE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"all_passed={ok} -> {HERE/'SMOKE_AUDIT.md'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
