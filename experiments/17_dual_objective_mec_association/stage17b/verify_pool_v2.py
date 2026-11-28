#!/usr/bin/env python3
"""Stage 17 Expert Pool V2 verification (V1..V10 of the review checklist)."""
import json, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
for p in [ROOT / "src", REPO / "experiments/16_cumulative_e2e_association/p1",
          REPO / "experiments/14_mec_expert_retraining",
          REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4/scripts",
          REPO / "experiments/06_closed_loop_success_rate/scripts"]:
    sys.path.insert(0, str(p))

from config17 import load_config
from env16 import build_deployment_by_map, load_norm
from env17 import EpisodeEnv17
from latency17 import d_inf_ms, service_latency_ms
from pool17 import KEYS, load_pool
from train_association_rl import load_tasks_and_split

B = 1
def batch(dev="cpu"):
    return {k: v.to(dev) for k, v in {
        "current": torch.randn(B, 49, 2048), "target_emb": torch.randn(B, 512),
        "history_vis": torch.randn(B, 16, 2048), "history_pose": torch.randn(B, 16, 3),
        "history_action": torch.randn(B, 16, 4), "history_mask": torch.ones(B, 16),
        "belief": torch.randn(B, 10), "match": torch.randn(B, 5),
        "landmarks": torch.randn(B, 8, 3), "states": torch.randn(B, 3)}.items()}

def main():
    dev = "cuda"; res = {}
    cfg = load_config(ROOT / "configs/default.json")
    pool = load_pool("v2", dev)

    # 1 / 3: all five load and produce 4 logits
    shapes, params = {}, {}
    with torch.no_grad():
        for k in KEYS:
            out = pool.models[k](batch(dev))
            shapes[k] = list(out.shape)
            params[k] = sum(p.numel() for p in pool.models[k].parameters())
    res["V1_load_and_shapes"] = {"shapes": shapes, "params": params,
                                 "pass": all(s == [1, 4] for s in shapes.values())}

    # 2: deployment
    dep = build_deployment_by_map()
    nodes = {m: sum(1 for n in dep[list(dep)[0]] if n["model_type"] == m) for m in KEYS}
    res["V2_deployment"] = {"maps": len(dep), "per_model": nodes,
                            "tflops": sorted({n["compute_tflops"] for n in dep[list(dep)[0]]}),
                            "pass": len(dep) > 0 and all(v == 2 for v in nodes.values())}

    # 4: old Small logits == direct Stage-11 construction
    from model11_small import SmallMEC4
    ck = torch.load(REPO / "experiments/11_teacher_geometry_ablation/11B_transformer_m_v4"
                    "/results/runs_small/seed42/best.pt", map_location=dev, weights_only=False)
    ref = SmallMEC4().to(dev).eval(); ref.load_state_dict(ck["model"])
    b = batch(dev)
    with torch.no_grad():
        a = pool.models["small"](b); r = ref(b)
    res["V4_small_equivalence"] = {"max_abs_diff": float((a - r).abs().max()),
                                   "pass": float((a - r).abs().max()) == 0.0}

    # 5: new TM logits == Stage-14 build_model + R3 checkpoint
    from models import build_model
    tm_ck = torch.load(REPO / "experiments/14_mec_expert_retraining/tm_r3_lr_continuation"
                       "/runs/tm/seed42/checkpoints/best.pt", map_location=dev, weights_only=False)
    tm = build_model("tm").to(dev).eval(); tm.load_state_dict(tm_ck["model"])
    b2 = batch(dev)
    with torch.no_grad():
        a2 = pool.models["tm"](b2); r2 = tm(b2)
    res["V5_tm_equivalence"] = {"max_abs_diff": float((a2 - r2).abs().max()),
                                "pass": float((a2 - r2).abs().max()) == 0.0,
                                "ckpt_epoch": tm_ck.get("epoch")}

    # 6: FLOPs / latency
    lat = {k: d_inf_ms(k, cfg, pool.flops) for k in KEYS}
    res["V6_latency"] = {"d_inf_ms": lat,
                         "small_expected": 0.004473344 / 10.0,
                         "pass": abs(lat["small"] - 0.0004473344) < 1e-12
                                 and abs(lat["tm"] - 10.0025418) < 1e-6}

    # 7-9: environment smoke with V2 (one task, feasible node)
    train_tasks, _ = load_tasks_and_split()
    env = EpisodeEnv17(build_deployment_by_map(), dev, pool.models, load_norm(),
                       cfg, flops_map=pool.flops)
    calls = {"expert": 0, "encoder": 0}

    class _C:
        """Callable proxy that preserves attributes (enc4.embed etc.)."""
        def __init__(self, obj, counter, key):
            self._obj, self._counter, self._key = obj, counter, key
        def __call__(self, *a, **kw):
            self._counter[self._key] += 1
            return self._obj(*a, **kw)
        def __getattr__(self, name):
            return getattr(self._obj, name)

    for k in KEYS:
        env.models[k] = _C(env.models[k], calls, "expert")
    env.enc4 = _C(env.enc4, calls, "encoder")
    env.start(train_tasks[0])
    feas = env.feasible_node_indices()
    _, done, info = env.step(feas[0] if feas else 0)
    res["V7_no_illegal_connection"] = {
        "feasible": len(feas), "failure": env.failure,
        "pass": env.terminated_reason not in ("invalid_mec_action", "no_feasible_mec")}
    res["V8_single_expert_forward"] = {"calls": calls["expert"],
                                       "pass": calls["expert"] == 1}
    res["V9_single_encoding"] = {"calls": calls["encoder"], "pass": calls["encoder"] == 1}
    res["V10_constraints_unchanged"] = {
        "d_max_ms": cfg.d_max_ms, "battery_j": cfg.task_energy_budget_j,
        "coverage_m": cfg.coverage_m,
        "pass": cfg.d_max_ms == 400.0 and cfg.task_energy_budget_j == 360000.0
                and cfg.coverage_m == 500.0}

    out = HERE / "pool_v2_verify.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    n = sum(1 for v in res.values() if v["pass"])
    print(f"PASSED {n}/{len(res)}")
    return 0 if n == len(res) else 1

if __name__ == "__main__":
    raise SystemExit(main())
