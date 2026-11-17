#!/usr/bin/env python3
"""Stage 13-R1: architecture parameter search with the unified input contract.

Transformer-S / Large use the canonical 104-token sequence (16 history tokens).
Small / Medium keep their internal pooling/cross-attention (allowed) and the same
modalities.  Targets stay 5/15/40/100/200 GFLOPs.

Outputs under revision_r1/results/:
  revised_candidate_architectures.csv
  revised_weight_transfer_audit.csv
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REV = HERE.parent
RES = REV / "results"

TARGET = {
    "Small": 5e9,
    "Medium": 15e9,
    "Transformer-S": 40e9,
    "Transformer-M": 100e9,
    "Large": 200e9,
}
T104 = 104
FF_MULT = {"Transformer-S": 3, "Transformer-M": 4, "Large": 4}


def flops(model, D, L=1, F=None, H=None, Fgc=None):
    if model == "Small":
        H = 2 * D if H is None else H
        return 2.0 * (4633 * D + (8 * D + 4) * H)
    if model == "Medium":
        H = 2 * D if H is None else H
        Fgc = 2 * D if Fgc is None else Fgc
        return 2.0 * (103035 * D + 104 * D * D + 2 * D * Fgc
                      + (7 * D + 4) * H)
    if model == "Transformer-M":
        F = 4 * D if F is None else F
        t = 104
        macs = 133804 * D + 53 * D * D
        macs += L * (4 * t * D * D + 2 * t * t * D + 2 * t * D * F)
        return 2.0 * macs
    # Transformer-S / Large, canonical 104-token input
    F = FF_MULT[model] * D if F is None else F
    Fgc = FF_MULT[model] * D if Fgc is None else Fgc
    t = T104
    macs = 133966 * D + 100 * D * D + 2 * D * Fgc
    macs += L * (4 * t * D * D + 2 * t * t * D + 2 * t * D * F)
    return 2.0 * macs


def params(model, D, L=1, F=None, H=None, Fgc=None):
    def lin(i, o):
        return i * o + o

    def ln(d):
        return 2 * d

    if model == "Small":
        H = 2 * D if H is None else H
        return (lin(2048, D) * 2 + lin(512, D) + lin(3, D) + lin(4, D)
                + lin(12, D) + lin(3, D) + lin(3, D)
                + lin(8 * D, H) + lin(H, 4))
    if model == "Medium":
        H = 2 * D if H is None else H
        Fgc = 2 * D if Fgc is None else Fgc
        p = lin(2048, D) + lin(512, D) + 4 * D * D + 4 * D
        p += ln(D) + lin(D, Fgc) + lin(Fgc, D) + ln(D)
        p += lin(2048, D) + lin(3, D) + lin(4, D) + lin(12, D)
        p += lin(3, D) + lin(3, D) + lin(7 * D, H) + lin(H, 4)
        return p
    if model == "Transformer-M":
        F = 4 * D if F is None else F
        p = lin(2048, D) + 49 * D + lin(512, D) + lin(2048, D)
        p += lin(3, D) + lin(D, D) + 4 * D + 16 * D
        p += lin(12, D) + lin(D, D) + lin(3, D) + lin(D, D)
        p += lin(3, D) + lin(D, D) + 7 * D + D + ln(D) + lin(D, 4)
        per = 4 * D * D + 4 * D + lin(D, F) + lin(F, D) + 4 * D
        return p + L * per
    # TS / Large: original formula + 20h (hist_time 16h + type_emb 7h - 3h)
    F = FF_MULT[model] * D if F is None else F
    Fgc = FF_MULT[model] * D if Fgc is None else Fgc
    p = lin(2048, D) + lin(512, D) + 4 * D * D + 4 * D
    p += ln(D) + lin(D, Fgc) + lin(Fgc, D) + ln(D)
    p += lin(2048, D) + lin(3, D) + lin(4, D) + lin(12, D)
    p += lin(3, D) + lin(3, D) + lin(D, 4)
    p += 20 * D
    per = 4 * D * D + 4 * D + lin(D, F) + lin(F, D) + 4 * D
    return p + L * per


def solve_quadratic(a, b, target):
    disc = b * b + 4 * a * target
    return (-b + math.sqrt(max(disc, 0.0))) / (2 * a)


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
    revised = {r["model"]: r
               for r in json.loads((RES / "revised_baseline_flops.json").read_text())}
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
        exp = revised[model]["flops_after"]
        assert abs(cur_flops - exp) / exp < 0.02, (
            model, cur_flops, exp)
        target = TARGET[model]
        D, L, F = s["D"], s.get("L", 1), s.get("F")
        H, Fgc = s.get("H"), s.get("Fgc")
        p0 = revised[model]["params_after"]

        # A1: FFN / fusion width only
        if model in ("Transformer-S", "Large"):
            proj = flops(model, D, L, 0)
            coeff = 2.0 * L * 2 * T104 * D
            needF = (target - proj) / coeff
            nf = flops(model, D, L, needF)
            rows.append(plan_row(model, "A1_ffn_width", "FFN width", F,
                                 needF, cur_flops, nf, needF <= 4 * F,
                                 "single FFN width; no +/-5% under 4x"))
            transfer.append((model, "A1_ffn_width", p0,
                             params(model, D, L, needF)))
        elif model == "Transformer-M":
            proj = flops(model, D, L, 0)
            coeff = 2.0 * L * 2 * 104 * D
            needF = (target - proj) / coeff
            nf = flops(model, D, L, needF)
            rows.append(plan_row(model, "A1_ffn_width", "FFN width", F,
                                 needF, cur_flops, nf, needF <= 4 * F,
                                 "single FFN width; no +/-5% under 4x"))
            transfer.append((model, "A1_ffn_width", p0,
                             params(model, D, L, needF)))
        elif model == "Medium":
            base_no = flops(model, D, H=H, Fgc=0)
            coeff = 2.0 * 2 * D
            needFgc = (target - base_no) / coeff
            nf = flops(model, D, H=H, Fgc=needFgc)
            rows.append(plan_row(model, "A1_gc_ffn_width", "gc_ffn width",
                                 Fgc, needFgc, cur_flops, nf, False,
                                 "single cross-attention FFN width"))
            transfer.append((model, "A1_gc_ffn_width", p0,
                             params(model, D, H=H, Fgc=needFgc)))
        else:
            base_no = flops(model, D, H=0)
            coeff = 2.0 * (8 * D + 4)
            needH = (target - base_no) / coeff
            nf = flops(model, D, H=needH)
            rows.append(plan_row(model, "A1_fusion_width", "fusion MLP width",
                                 H, needH, cur_flops, nf, False,
                                 "single late-fusion MLP width"))
            transfer.append((model, "A1_fusion_width", p0,
                             params(model, D, H=needH)))

        # A2: layers only
        if model in ("Transformer-S", "Transformer-M", "Large"):
            per = flops(model, D, 1, F) - flops(model, D, 0, F)
            proj = flops(model, D, 0, F)
            needL = (target - proj) / per
            nf = flops(model, D, needL, F)
            rows.append(plan_row(model, "A2_layers", "encoder layers", L,
                                 needL, cur_flops, nf, needL <= 4 * L,
                                 "single encoder-layer count; no +/-5% under 4x"))
            transfer.append((model, "A2_layers", p0,
                             params(model, D, needL, F)))
        else:
            rows.append(plan_row(model, "A2_layers", "encoder layers", 0, 0,
                                 cur_flops, cur_flops, False,
                                 "no Transformer encoder in this model"))

        # A3: hidden dimension only
        if model == "Small":
            needD = solve_quadratic(32.0, 9282.0, target)
        elif model == "Medium":
            # H=2D, Fgc=2D: FLOPs = 244 D^2 + 206098 D
            needD = solve_quadratic(244.0, 206098.0, target)
        elif model == "Transformer-M":
            a = 2.0 * (53 + L * (4 * 104 + 8 * 104))
            b = 2.0 * (133804 + L * 2 * 104 * 104)
            needD = solve_quadratic(a, b, target)
        else:
            a = 2.0 * (100 + L * (4 * T104 + 2 * T104 * FF_MULT[model]))
            b = 2.0 * (133966 + L * 2 * T104 * T104)
            needD = solve_quadratic(a, b, target)
        nf = flops(model, needD, L, None)
        rows.append(plan_row(model, "A3_hidden_dim", "hidden dim", D, needD,
                             cur_flops, nf, False,
                             "hidden width only; large parameter/width growth"))
        transfer.append((model, "A3_hidden_dim", p0,
                         params(model, needD, L, None)))

        # B: best two-parameter under 4x caps
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
                f"best under F<=4x,L<=4x; {100*fl/target:.2f}% of target"))
            transfer.append((model, "B_two_param_cap", p0,
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
                f"best under D<=2x,H<=4x; {100*fl/target:.2f}% of target"))
            transfer.append((model, "B_two_param_cap", p0,
                             params(model, D * dm, H=H * hm, Fgc=Fgc * hm)))
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
                f"best under D<=2x,H<=4x; {100*fl/target:.2f}% of target"))
            transfer.append((model, "B_two_param_cap", p0,
                             params(model, D * dm, H=H * hm)))

        # C: target-hitting multi-parameter (hidden)
        if model in ("Transformer-S", "Transformer-M", "Large"):
            cD = needD
            cF = FF_MULT[model] * cD
            cfl = flops(model, cD, L, cF)
            rows.append(plan_row(
                model, "C_target_multi", "D (F=ff_mult*D)", D, cD,
                cur_flops, cfl, False,
                f"target needs D~{cD:.0f} ({cD/D:.1f}x); params "
                f"~{params(model, cD, L, cF)/1e6:.0f}M"))
            transfer.append((model, "C_target_multi", p0,
                             params(model, cD, L, cF)))
        elif model == "Medium":
            cD = solve_quadratic(244.0, 206098.0, target)
            cfl = flops(model, cD)
            rows.append(plan_row(model, "C_target_multi", "D (Fgc=2D,H=2D)",
                                 D, cD, cur_flops, cfl, False,
                                 f"target needs D~{cD:.0f} ({cD/D:.1f}x)"))
            transfer.append((model, "C_target_multi", p0,
                             params(model, cD)))
        else:
            cD = solve_quadratic(32.0, 9282.0, target)
            cfl = flops(model, cD)
            rows.append(plan_row(model, "C_target_multi", "D (H=2D)", D, cD,
                                 cur_flops, cfl, False,
                                 f"target needs D~{cD:.0f} ({cD/D:.1f}x)"))
            transfer.append((model, "C_target_multi", p0,
                             params(model, cD)))

    with (RES / "revised_candidate_architectures.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    tr_rows = []
    for model, plan, p_old, p_new in transfer:
        p_new = float(p_new)
        if plan.startswith("A1") or plan.startswith("B_"):
            changed_old = 0.25 * p_old
            exact = p_old - changed_old
            partial = changed_old
            new = max(p_new - p_old, 0.0)
        elif plan.startswith("A2"):
            exact = p_old
            partial = 0.0
            new = max(p_new - p_old, 0.0)
        elif plan.startswith("A3") or plan.startswith("C_"):
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
    with (RES / "revised_weight_transfer_audit.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tr_rows[0].keys()),
                           lineterminator="\n")
        w.writeheader()
        w.writerows(tr_rows)
    print(f"wrote {RES/'revised_candidate_architectures.csv'}")
    print(f"wrote {RES/'revised_weight_transfer_audit.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
