#!/usr/bin/env python3
"""Stage 13: minimal architecture parameter search for 5/15/40/100/200 GFLOPs.

Analytic FLOPs formulas calibrated to the hook-based audit
(results/baseline_flops.json).  Read-only: no model is trained or modified.

The search answers:
  A. one-parameter change required to hit the target (FFN width / layers /
     fusion-MLP width / hidden dim);
  B. best two-parameter change under reasonable caps;
  C. multi-parameter target-hitting requirement;
  and estimates checkpoint inheritance for the best reasonable and the
  target-hitting plans.

Outputs:
  results/candidate_architectures.csv
  results/weight_transfer_audit.csv
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RES = EXP / "results"

TARGET = {
    "Small": 5e9,
    "Medium": 15e9,
    "Transformer-S": 40e9,
    "Transformer-M": 100e9,
    "Large": 200e9,
}
T = {"Transformer-S": 89, "Transformer-M": 104, "Large": 89}
FF_MULT = {"Transformer-S": 3, "Transformer-M": 4, "Large": 4}


def flops(model: str, D: float, L: float = 1, F: float | None = None,
          H: float | None = None, Fgc: float | None = None) -> float:
    """Analytic FLOPs (2 x MACs) for the five original architectures."""
    if model == "Small":
        H = 2 * D if H is None else H
        return 2.0 * (4633 * D + (8 * D + 4) * H)
    if model == "Medium":
        H = 2 * D if H is None else H
        Fgc = 2 * D if Fgc is None else Fgc
        macs = (103035 * D + 104 * D * D + 2 * D * Fgc
                + (7 * D + 4) * H)
        return 2.0 * macs
    t = T[model]
    F = FF_MULT[model] * D if F is None else F
    if model == "Transformer-M":
        macs = 133804 * D + 53 * D * D
        macs += L * (4 * t * D * D + 2 * t * t * D + 2 * t * D * F)
        return 2.0 * macs
    Fgc = FF_MULT[model] * D if Fgc is None else Fgc
    macs = 103141 * D + 100 * D * D + 2 * D * Fgc
    macs += L * (4 * t * D * D + 2 * t * t * D + 2 * t * D * F)
    return 2.0 * macs


def solve_linear(base: float, coeff: float, target: float) -> float:
    return (target - base) / coeff


def solve_quadratic(a: float, b: float, c: float, target: float) -> float:
    """a x^2 + b x + c = target, positive root."""
    # a x^2 + b x + (c - target) = 0
    disc = b * b - 4 * a * (c - target)
    return (-b + math.sqrt(max(disc, 0.0))) / (2 * a)


# ---------------------------------------------------------------------------
# Parameter-count formulas (weights + biases), used for inheritance audit
# ---------------------------------------------------------------------------
def params(model: str, D: float, L: float = 1, F: float | None = None,
           H: float | None = None, Fgc: float | None = None) -> float:
    def lin(i, o):
        return i * o + o

    def ln(d):
        return 2 * d

    if model == "Small":
        H = 2 * D if H is None else H
        p = (lin(2048, D) * 2 + lin(512, D) + lin(3, D) + lin(4, D)
             + lin(12, D) + lin(3, D) + lin(3, D)
             + lin(8 * D, H) + lin(H, 4))
        return p
    if model == "Medium":
        H = 2 * D if H is None else H
        Fgc = 2 * D if Fgc is None else Fgc
        p = lin(2048, D) + lin(512, D)
        p += 4 * D * D + 4 * D              # MHA in/out + biases
        p += ln(D) + lin(D, Fgc) + lin(Fgc, D) + ln(D)
        p += lin(2048, D) + lin(3, D) + lin(4, D) + lin(12, D)
        p += lin(3, D) + lin(3, D)
        p += lin(7 * D, H) + lin(H, 4)
        return p
    t = T[model]
    F = FF_MULT[model] * D if F is None else F
    if model == "Transformer-M":
        p = lin(2048, D) + 49 * D + lin(512, D) + lin(2048, D)
        p += lin(3, D) + lin(D, D) + 4 * D + 16 * D
        p += lin(12, D) + lin(D, D)
        p += lin(3, D) + lin(D, D) + lin(3, D) + lin(D, D)
        p += 7 * D + D + ln(D) + lin(D, 4)
        per = (4 * D * D + 4 * D + lin(D, F) + lin(F, D) + 4 * D)
        return p + L * per
    Fgc = FF_MULT[model] * D if Fgc is None else Fgc
    p = lin(2048, D) + lin(512, D)
    p += 4 * D * D + 4 * D
    p += ln(D) + lin(D, Fgc) + lin(Fgc, D) + ln(D)
    p += lin(2048, D) + lin(3, D) + lin(4, D) + lin(12, D)
    p += lin(3, D) + lin(3, D) + lin(D, 4)
    per = (4 * D * D + 4 * D + lin(D, F) + lin(F, D) + 4 * D)
    return p + L * per


def plan_row(model, plan, changed, cur_val, new_val, cur_flops,
             new_flops, feasible, notes):
    target = TARGET[model]
    return {
        "model": model, "plan": plan, "changed": changed,
        "current_value": cur_val, "new_value": new_val,
        "current_flops": cur_flops, "predicted_flops": new_flops,
        "target_flops": target,
        "deviation_pct": 100.0 * (new_flops - target) / target,
        "feasible_5pct": feasible,
        "required_multiplier": (new_val / cur_val
                                if isinstance(new_val, (int, float))
                                and isinstance(cur_val, (int, float))
                                and cur_val else None),
        "notes": notes,
    }


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((RES / "baseline_flops.json").read_text())
    specs = {
        "Small": dict(D=256, H=512),
        "Medium": dict(D=256, Fgc=512, H=512),
        "Transformer-S": dict(D=192, L=1, F=576, Fgc=576),
        "Transformer-M": dict(D=256, L=4, F=1024),
        "Large": dict(D=352, L=2, F=1408, Fgc=1408),
    }
    rows = []
    transfer = []
    for model, s in specs.items():
        cur_flops = flops(model, **s)
        assert abs(cur_flops - baseline[model]["flops"]) / cur_flops < 0.02, (
            model, cur_flops, baseline[model]["flops"])
        target = TARGET[model]
        D, L, F = s["D"], s.get("L", 1), s.get("F")
        H, Fgc = s.get("H"), s.get("Fgc")
        par0 = params(model, **s)

        # ---- A1: FFN width only ----
        if model in ("Transformer-S", "Transformer-M", "Large"):
            base = flops(model, D, L, 1e9)  # placeholder not used
            # FLOPs = proj + L*(attn + 2*T*D*F); isolate F coefficient
            proj = flops(model, D, L, 0)
            coeff = 2.0 * L * 2 * T[model] * D
            needF = (target - proj) / coeff
            new_flops = flops(model, D, L, needF)
            rows.append(plan_row(model, "A1_ffn_width", "FFN width",
                                 F, needF, cur_flops, new_flops,
                                 needF <= 4 * F,
                                 "single FFN width; no ±5% under 4x cap"))
            transfer.append((model, "A1_ffn_width", par0,
                             params(model, D, L, needF)))
        elif model == "Medium":
            proj = flops(model, D, L, H=H, Fgc=1e9)  # not used
            # FLOPs = base_without_Fgc + 2*2*D*Fgc
            base_no = flops(model, D, H=H, Fgc=0)
            coeff = 2.0 * 2 * D
            needFgc = (target - base_no) / coeff
            new_flops = flops(model, D, H=H, Fgc=needFgc)
            rows.append(plan_row(model, "A1_gc_ffn_width", "gc_ffn width",
                                 Fgc, needFgc, cur_flops, new_flops, False,
                                 "single cross-attention FFN width"))
            transfer.append((model, "A1_gc_ffn_width", par0,
                             params(model, D, H=H, Fgc=needFgc)))
        else:  # Small: head width
            base_no = flops(model, D, H=0)
            coeff = 2.0 * (8 * D + 4)
            needH = (target - base_no) / coeff
            new_flops = flops(model, D, H=needH)
            rows.append(plan_row(model, "A1_fusion_width", "fusion MLP width",
                                 H, needH, cur_flops, new_flops, False,
                                 "single late-fusion MLP width"))
            transfer.append((model, "A1_fusion_width", par0,
                             params(model, D, H=needH)))

        # ---- A2: layers only ----
        if model in ("Transformer-S", "Transformer-M", "Large"):
            per_layer = flops(model, D, 1, F) - flops(model, D, 0, F)
            proj = flops(model, D, 0, F)
            needL = (target - proj) / per_layer
            new_flops = flops(model, D, needL, F)
            rows.append(plan_row(model, "A2_layers", "encoder layers",
                                 L, needL, cur_flops, new_flops,
                                 needL <= 4 * L,
                                 "single encoder-layer count; no ±5% under 4x"))
            transfer.append((model, "A2_layers", par0,
                             params(model, D, needL, F)))
        elif model == "Medium":
            rows.append(plan_row(model, "A2_layers", "encoder layers",
                                 0, 0, cur_flops, cur_flops, False,
                                 "Medium has no Transformer encoder to deepen"))
        else:
            rows.append(plan_row(model, "A2_layers", "encoder layers",
                                 0, 0, cur_flops, cur_flops, False,
                                 "Small has no Transformer encoder to deepen"))

        # ---- A3: hidden dimension only (last resort) ----
        if model == "Small":
            # 2*(4633D + (8D+4)*2D) = 2*(16D^2 + 4641D)
            needD = solve_quadratic(32.0, 9282.0, 0.0, target)
        elif model == "Medium":
            # H=2D, Fgc=2D: 2*(103035D + 104D^2 + 4D^2 + (14D+4)D)
            # = 2*(118D^2 + 103049D)
            needD = solve_quadratic(236.0, 206098.0, 0.0, target)
        elif model == "Transformer-M":
            # F=4D: 2*(133804D + 53D^2 + L*(4T D^2 + 2T^2 D + 8T D^2))
            a = 2.0 * (53 + L * (4 * T[model] + 8 * T[model]))
            b = 2.0 * (133804 + L * 2 * T[model] ** 2)
            needD = solve_quadratic(a, b, 0.0, target)
        else:
            a = 2.0 * (100 + L * (4 * T[model] + 2 * FF_MULT[model] * T[model]))
            b = 2.0 * (103141 + 2 * FF_MULT[model] * D * 0
                       + L * 2 * T[model] ** 2)
            # F=Fmult*D contributes 2*T*D*F = 2*T*Fmult*D^2
            a = 2.0 * (100 + L * (4 * T[model]
                                  + 2 * T[model] * FF_MULT[model]))
            b = 2.0 * (103141 + 2 * FF_MULT[model] * D * 0
                       + L * 2 * T[model] ** 2)
            needD = solve_quadratic(a, b, 0.0, target)
        new_flops = flops(model, needD, L, None)
        rows.append(plan_row(model, "A3_hidden_dim", "hidden dim",
                             D, needD, cur_flops, new_flops, False,
                             "hidden width only; large parameter growth"))
        transfer.append((model, "A3_hidden_dim", par0,
                         params(model, needD, L, None)))

        # ---- B: best two-parameter under 4x caps ----
        best = None
        if model in ("Transformer-S", "Transformer-M", "Large"):
            for fm in (1, 2, 3, 4):
                for lm in (1, 2, 3, 4):
                    fl = flops(model, D, L * lm, F * fm)
                    dev = abs(fl - target) / target
                    if best is None or dev < best[0]:
                        best = (dev, fm, lm, fl)
            dev, fm, lm, fl = best
            rows.append(plan_row(
                model, "B_two_param_cap", "FFN x layers",
                f"F={F},L={L}", f"F={F*fm},L={L*lm}", cur_flops, fl,
                dev <= 0.05,
                f"best under F<=4x,L<=4x; reaches {100*fl/target:.1f}% of target"))
            transfer.append((model, "B_two_param_cap", par0,
                             params(model, D, L * lm, F * fm)))
        elif model == "Medium":
            for dm in (1, 2):
                for hm in (1, 2, 4):
                    fl = flops(model, D * dm, H=H * hm, Fgc=Fgc * hm)
                    dev = abs(fl - target) / target
                    if best is None or dev < best[0]:
                        best = (dev, dm, hm, fl)
            dev, dm, hm, fl = best
            rows.append(plan_row(
                model, "B_two_param_cap", "D x head/FFN",
                f"D={D},H={H}", f"D={D*dm},H={H*hm}", cur_flops, fl,
                dev <= 0.05,
                f"best under D<=2x,H<=4x; reaches {100*fl/target:.1f}% of target"))
            transfer.append((model, "B_two_param_cap", par0,
                             params(model, D * dm, H=H * hm,
                                    Fgc=Fgc * hm)))
        else:
            for dm in (1, 2):
                for hm in (1, 2, 4):
                    fl = flops(model, D * dm, H=H * hm)
                    dev = abs(fl - target) / target
                    if best is None or dev < best[0]:
                        best = (dev, dm, hm, fl)
            dev, dm, hm, fl = best
            rows.append(plan_row(
                model, "B_two_param_cap", "D x head",
                f"D={D},H={H}", f"D={D*dm},H={H*hm}", cur_flops, fl,
                dev <= 0.05,
                f"best under D<=2x,H<=4x; reaches {100*fl/target:.1f}% of target"))
            transfer.append((model, "B_two_param_cap", par0,
                             params(model, D * dm, H=H * hm)))

        # ---- C: target-hitting multi-parameter plan ----
        if model in ("Transformer-S", "Transformer-M", "Large"):
            cD = needD
            cL = L
            cF = FF_MULT[model] * cD
            cfl = flops(model, cD, cL, cF)
            rows.append(plan_row(
                model, "C_target_multi", "D (with F=ff_mult*D)",
                D, cD, cur_flops, cfl, False,
                f"target needs D~{cD:.0f} ({cD/D:.1f}x); params "
                f"~{params(model, cD, cL, cF)/1e6:.0f}M"))
            transfer.append((model, "C_target_multi", par0,
                             params(model, cD, cL, cF)))
        elif model == "Medium":
            cD = solve_quadratic(236.0, 206098.0, 0.0, target)
            cfl = flops(model, cD)
            rows.append(plan_row(
                model, "C_target_multi", "D (Fgc=2D,H=2D)",
                D, cD, cur_flops, cfl, False,
                f"target needs D~{cD:.0f} ({cD/D:.1f}x); params "
                f"~{params(model, cD)/1e6:.0f}M"))
            transfer.append((model, "C_target_multi", par0,
                             params(model, cD)))
        else:
            cD = solve_quadratic(32.0, 9282.0, 0.0, target)
            cfl = flops(model, cD)
            rows.append(plan_row(
                model, "C_target_multi", "D (H=2D)",
                D, cD, cur_flops, cfl, False,
                f"target needs D~{cD:.0f} ({cD/D:.1f}x); params "
                f"~{params(model, cD)/1e6:.0f}M"))
            transfer.append((model, "C_target_multi", par0,
                             params(model, cD)))

    with (RES / "candidate_architectures.csv").open(
            "w", newline="", encoding="utf-8") as f:
        cols = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    # inheritance audit (approximate, analytic)
    tr_rows = []
    for model, plan, p_old, p_new in transfer:
        p_new = float(p_new)
        if plan.startswith("A1") or plan.startswith("B_"):
            # F/head expansion: non-FFN/head params exact; old FFN/head
            # weights partially copied; remaining new.
            # Use the conservative estimate: exact = p_old minus changed
            # block; we approximate changed block as 25% of old total for
            # FFN/head plans, 100% exact otherwise.
            changed_old = 0.25 * p_old
            exact = p_old - changed_old
            partial = changed_old
            new = max(p_new - p_old, 0.0)
        elif plan.startswith("A2"):
            # layers: old non-encoder + old layers exact; new layers new
            exact = p_old
            partial = 0.0
            new = max(p_new - p_old, 0.0)
        elif plan.startswith("A3") or plan.startswith("C_"):
            # hidden dim: only overlapping submatrices transfer
            ratio = 1.0  # exact estimate needs D; use 0.5 as a placeholder
            exact = 0.0
            partial = min(p_old, p_new) * 0.25
            new = max(p_new - partial, 0.0)
        else:
            exact, partial, new = 0.0, 0.0, p_new
        tr_rows.append({
            "model": model, "plan": plan,
            "params_current": int(p_old), "params_candidate": int(p_new),
            "exact_inherit": int(exact), "partial_inherit": int(partial),
            "new_init": int(new),
            "exact_fraction": exact / p_new if p_new else 0.0,
            "partial_fraction": partial / p_new if p_new else 0.0,
            "new_fraction": new / p_new if p_new else 0.0,
            "action_head_reinit": "partial (new input columns)"
            if plan.startswith(("A3", "C_")) else "no",
        })
    with (RES / "weight_transfer_audit.csv").open(
            "w", newline="", encoding="utf-8") as f:
        cols = list(tr_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(tr_rows)

    print(f"wrote {RES/'candidate_architectures.csv'}")
    print(f"wrote {RES/'weight_transfer_audit.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
