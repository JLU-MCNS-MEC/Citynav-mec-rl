#!/usr/bin/env python3
"""Stage 14-R1 live report: PROGRESS_REPORT.md + figures.

Reads results/<Model>/ (training history, closed-loop summary) and writes
  PROGRESS_REPORT.md (also results/PROGRESS_REPORT.md)
  figures/{sr_by_model,loss_curves,dev_macro_f1,compute_vs_params}.png
Never fabricates numbers: unfinished models stay Pending.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KEYS = ["small", "medium", "ts", "tm", "large"]
NAMES = {"small": "Small", "medium": "Medium", "ts": "Transformer-S",
         "tm": "Transformer-M", "large": "Large"}
TARGETS = {"small": 5, "medium": 15, "ts": 40, "tm": 100, "large": 200}
PARAMS = {"small": 61_121_028, "medium": 166_011_908, "ts": 211_181_060,
          "tm": 498_567_684, "large": 1_034_877_444}
R3_EPOCH_H = {"small": 0.296, "medium": 0.566, "ts": 0.836,
              "tm": 1.139, "large": 1.862}
EXPECTED_EPOCHS = 15          # early-stop expectation; replaced by actual history
EVAL_H = 0.6                  # ~2498 tasks at ~1.2-1.5 ep/s


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def model_state(key: str) -> dict:
    name = NAMES[key]
    rdir = HERE / "results" / name
    hist = load_json(rdir / "training_history.json") or []
    summary = load_json(rdir / "closed_loop/val_seen_summary.json")
    progress = load_json(rdir / "train_progress.json")
    cfg = load_json(rdir / "config.json")
    f1s = [h["dev"]["macro_f1"] for h in hist]
    best_epoch = int(np.argmax(f1s)) if f1s else None
    secs = [float(h.get("seconds", 0)) for h in hist]
    epoch_sec = float(np.mean(secs)) if secs else None
    return {"key": key, "name": name, "hist": hist, "summary": summary,
            "progress": progress, "config": cfg, "best_epoch": best_epoch,
            "best_f1": f1s[best_epoch] if f1s else None,
            "train_h": sum(secs) / 3600 if secs else None,
            "epoch_sec": epoch_sec, "params": PARAMS[key],
            "target": TARGETS[key],
            "peak_gb": max((float(h.get("peak_mem_gb", 0)) for h in hist),
                           default=None)}


def fmt(v, p=4, suffix=""):
    return "Pending" if v is None else f"{v:.{p}f}{suffix}"


def num(v, p=4, suffix=""):
    return "—" if v is None else f"{v:.{p}f}{suffix}"


def figures(states: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"WARN matplotlib unavailable: {e}")
        return
    figdir = HERE / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    # SR bar
    plt.figure(figsize=(7, 4))
    labels = [s["name"] for s in states]
    vals = [s["summary"]["sr"] if s["summary"] else 0.0 for s in states]
    done = [s["summary"] is not None for s in states]
    colors = ["#2a6fdb" if d else "#cccccc" for d in done]
    bars = plt.bar(labels, vals, color=colors)
    for b, s, d in zip(bars, states, done):
        txt = f"{s['summary']['sr']*100:.1f}%" if d else "Pending"
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, txt,
                 ha="center", fontsize=9)
    plt.ylim(0, 1); plt.ylabel("Val-Seen SR (2498 tasks)")
    plt.title("Stage 14 MEC experts: closed-loop SR"); plt.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(figdir / "sr_by_model.png", dpi=130); plt.close()
    # Loss curves
    plt.figure(figsize=(8, 4.5))
    for s in states:
        if s["hist"]:
            ep = [h["epoch"] + 1 for h in s["hist"]]
            plt.plot(ep, [h["train_loss"] for h in s["hist"]], marker="o",
                     ms=3, label=f"{s['name']} train")
            plt.plot(ep, [h["dev"]["loss"] for h in s["hist"]], linestyle="--",
                     lw=1, label=f"{s['name']} dev")
    plt.xlabel("epoch"); plt.ylabel("cross-entropy"); plt.grid(alpha=.3)
    if any(s["hist"] for s in states):
        plt.legend(fontsize=7, ncol=2)
    plt.title("Training / Dev loss")
    plt.tight_layout(); plt.savefig(figdir / "loss_curves.png", dpi=130); plt.close()
    # Dev Macro-F1
    plt.figure(figsize=(8, 4.5))
    for s in states:
        if s["hist"]:
            ep = [h["epoch"] + 1 for h in s["hist"]]
            plt.plot(ep, [h["dev"]["macro_f1"] for h in s["hist"]], marker="o",
                     ms=3, label=s["name"])
    plt.xlabel("epoch"); plt.ylabel("Dev Macro-F1"); plt.grid(alpha=.3)
    if any(s["hist"] for s in states):
        plt.legend(fontsize=8)
    plt.title("Independent Dev Macro-F1 (2% hold-out)")
    plt.tight_layout(); plt.savefig(figdir / "dev_macro_f1.png", dpi=130); plt.close()
    # Compute vs params
    plt.figure(figsize=(6, 4.5))
    xs = [s["params"] / 1e6 for s in states]
    ys = [s["target"] for s in states]
    plt.scatter(xs, ys, s=90, c="#2a6fdb")
    for x, y, s in zip(xs, ys, states):
        plt.annotate(s["name"], (x, y), textcoords="offset points", xytext=(6, 5),
                     fontsize=8)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("parameters (M, log)"); plt.ylabel("target GFLOPs (log)")
    plt.grid(alpha=.3, which="both"); plt.title("Compute vs parameters")
    plt.tight_layout(); plt.savefig(figdir / "compute_vs_params.png", dpi=130)
    plt.close()


def estimates(states: list[dict]) -> tuple[float | None, str]:
    """Remaining training hours from actual epoch times (R3 fallback)."""
    remaining = 0.0
    lines = []
    for s in states:
        if s["summary"] is not None:
            continue
        if s["hist"]:
            ep = len(s["hist"])
            epochs = EXPECTED_EPOCHS if ep < EXPECTED_EPOCHS else ep
            left = max(0, epochs - ep)
            h = (s["epoch_sec"] or 0) * left / 3600
            remaining += h
            lines.append(f"{s['name']}: {left} epochs × "
                         f"{(s['epoch_sec'] or 0)/3600:.2f} h = {h:.1f} h")
        else:
            h = R3_EPOCH_H[s["key"]] * EXPECTED_EPOCHS
            remaining += h
            lines.append(f"{s['name']}: R3 {R3_EPOCH_H[s['key']]:.2f} h/ep × "
                         f"{EXPECTED_EPOCHS} = {h:.1f} h")
    return remaining, "; ".join(lines)


def live_gpu() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,"
             "temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        mem, util, temp = out.split(",")
        return {"mem_mib": int(mem.strip().split()[0]),
                "gpu_util_pct": int(util.strip().split()[0]),
                "temp_c": int(temp.strip().split()[0])}
    except Exception:  # noqa: BLE001
        return {}


def training_running(key: str) -> bool:
    try:
        return subprocess.run(
            ["pgrep", "-f", f"train_expert.py --model {key}"],
            capture_output=True).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def latest_update_line(key: str) -> dict | None:
    p = HERE / "logs" / f"train_{key}.log"
    if not p.exists():
        return None
    last = None
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"epoch (\d+) update (\d+)/(\d+) loss=([0-9.]+) "
                      r"acc=([0-9.]+) (\d+)s", line)
        if m:
            last = {"epoch": int(m.group(1)), "update": int(m.group(2)),
                    "updates": int(m.group(3)), "loss": float(m.group(4)),
                    "acc": float(m.group(5)), "elapsed": int(m.group(6))}
    return last


def _bold_best(vals, higher: bool) -> list[str]:
    present = [v for v in vals if v is not None]
    if len(present) < 2:
        return ["—" if v is None else f"{v:.4f}" for v in vals]
    best = max(present) if higher else min(present)
    return ["—" if v is None else
            (f"**{v:.4f}**" if v == best else f"{v:.4f}") for v in vals]


def comparison_report(states: list[dict]) -> None:
    """Cross-model comparison: closed loop, training summary, per-epoch curves."""
    md = ["# Stage 14 五模型对比", "",
          f"更新时间：{time.strftime('%Y-%m-%d %H:%M:%S')}", "",
          "五种模型使用相同 seed、相同训练 episode 顺序、相同 2% 独立 Dev 集，",
          "因此**相同 epoch 编号可直接横向对比**。未完成项显示 Pending / —。", ""]

    # 1) closed-loop navigation comparison
    md += ["## 1. Val-Seen 2498 闭环导航指标", "",
           "| 模型 | SR | OSR | SPL | NE (m) | nDTW | 平均步数 | Stop成功率 | 主要失败模式 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for s in states:
        sm = s["summary"]
        if sm:
            fails = sorted(((k, v) for k, v in
                            (sm.get("failure_taxonomy") or {}).items()
                            if k != "success"), key=lambda x: -x[1])[:2]
            ftxt = "；".join(f"{k} {v}" for k, v in fails) or "—"
            md.append(
                f"| {s['name']} | {sm['sr']:.4f} | {sm['osr']:.4f} | "
                f"{sm['spl']:.4f} | {sm['ne_m']:.2f} | "
                f"{sm['ndtw']:.4f} | {sm['mean_steps']:.1f} | "
                f"{sm['stop_success_rate']:.4f} | {ftxt} |")
        else:
            md.append(f"| {s['name']} | Pending | Pending | Pending | Pending | "
                      f"Pending | Pending | Pending | — |")
    md += ["", "SR 与 OSR 以 20 m 半径为成功；NE 越小越好；SPL/nDTW 越大越好。", ""]

    # 2) capacity + training summary
    md += ["## 2. 容量与训练结果", "",
           "| 模型 | 目标FLOPs | 参数量 | Best Epoch | Best Dev F1 | Epochs | "
           "Train Time | 峰值显存 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in states:
        bx = "Pending" if s["best_epoch"] is None else str(s["best_epoch"] + 1)
        md.append(
            f"| {s['name']} | {s['target']} G | {s['params']/1e6:.1f} M | {bx} | "
            f"{fmt(s['best_f1'])} | {len(s['hist'])} | "
            f"{fmt(s['train_h'], 2, ' h')} | {fmt(s['peak_gb'], 2, ' GB')} |")
    md += [""]

    max_ep = max((len(s["hist"]) for s in states), default=0)
    if max_ep:
        # 3) per-epoch Dev Macro-F1
        md += ["## 3. 每个 Epoch 的 Dev Macro-F1（跨模型同 epoch 对比）", "",
               "| Epoch | " + " | ".join(s["name"] for s in states) + " |",
               "|---:|" + "---:|" * len(states)]
        for i in range(max_ep):
            vals = [s["hist"][i]["dev"]["macro_f1"] if i < len(s["hist"]) else None
                    for s in states]
            md.append(f"| {i+1} | " + " | ".join(_bold_best(vals, True)) + " |")
        md += [""]
        # 4) per-epoch Train Loss
        md += ["## 4. 每个 Epoch 的 Train Loss（跨模型同 epoch 对比）", "",
               "| Epoch | " + " | ".join(s["name"] for s in states) + " |",
               "|---:|" + "---:|" * len(states)]
        for i in range(max_ep):
            vals = [s["hist"][i]["train_loss"] if i < len(s["hist"]) else None
                    for s in states]
            md.append(f"| {i+1} | " + " | ".join(_bold_best(vals, False)) + " |")
        md += [""]
        # 5) per-epoch Dev Loss
        md += ["## 5. 每个 Epoch 的 Dev Loss（跨模型同 epoch 对比）", "",
               "| Epoch | " + " | ".join(s["name"] for s in states) + " |",
               "|---:|" + "---:|" * len(states)]
        for i in range(max_ep):
            vals = [s["hist"][i]["dev"]["loss"] if i < len(s["hist"]) else None
                    for s in states]
            md.append(f"| {i+1} | " + " | ".join(_bold_best(vals, False)) + " |")
        md += [""]

    md += ["## 说明", "",
           "- 同 epoch 编号对比是公平的：所有模型共享同一数据顺序与同一 Dev 集。",
           "- 训练 loss 只反映拟合能力，最终效果以闭环 SR/OSR/SPL/NE/nDTW 为准。",
           "- 新模型闭环完成后，第 1 节会自动加入该模型，便于直接横向比较。", ""]
    text = "\n".join(md)
    (HERE / "MODEL_COMPARISON.md").write_text(text, encoding="utf-8")
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    (HERE / "results/MODEL_COMPARISON.md").write_text(text, encoding="utf-8")

    # long-format CSV for analysis
    rows = ["model,epoch,train_loss,train_acc,dev_loss,dev_acc,dev_macro_f1,"
            "seconds,peak_mem_gb"]
    for s in states:
        for h in s["hist"]:
            d = h["dev"]
            rows.append(f"{s['name']},{h['epoch']+1},{h['train_loss']:.6f},"
                        f"{h['train_acc']:.6f},{d['loss']:.6f},{d['accuracy']:.6f},"
                        f"{d['macro_f1']:.6f},{h.get('seconds', 0)},"
                        f"{h.get('peak_mem_gb', 0):.4f}")
    (HERE / "results/epoch_comparison.csv").write_text("\n".join(rows) + "\n",
                                                       encoding="utf-8")


def main() -> int:
    # The eval venv has no matplotlib; re-exec under the training env.
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        alt = "/home/jlu-mcns/miniconda3/envs/citynav-mec/bin/python"
        if os.path.exists(alt) and os.path.realpath(sys.executable) != \
                os.path.realpath(alt):
            os.execv(alt, [alt, __file__] + sys.argv[1:])
    states = [model_state(k) for k in KEYS]
    figures(states)
    comparison_report(states)
    cur = None
    for s in states:                     # live process takes precedence
        if training_running(s["key"]):
            cur = s
            break
    if cur is None:
        for s in states:
            if s["progress"] and not s["progress"].get("done"):
                cur = s
                break
    live = latest_update_line(cur["key"]) if cur else None
    other = [s for s in states if cur is None or s["key"] != cur["key"]]
    other_h, other_text = estimates(other)
    eta_cur = None
    epoch_est_s = None
    if cur:
        p = cur["progress"] or {}
        if live and live["update"] > 0:
            per_upd = live["elapsed"] / live["update"]
            epoch_est_s = per_upd * live["updates"]
            left_in_epoch = max(0.0, epoch_est_s - live["elapsed"])
            left_epochs = max(0, EXPECTED_EPOCHS - (live["epoch"] + 1))
            eta_cur = (left_in_epoch + left_epochs * epoch_est_s) / 3600
        elif cur["epoch_sec"]:
            epoch_est_s = cur["epoch_sec"]
            left = max(0, EXPECTED_EPOCHS - (p.get("epoch", 0) + 1))
            eta_cur = left * epoch_est_s / 3600
    if eta_cur is not None:
        remain_h = eta_cur + other_h
        estimate_text = (f"{cur['name']}（进行中）{eta_cur:.1f} h; "
                         + (other_text or ""))
    else:
        remain_h, estimate_text = estimates(states)
    lines = ["# Stage 14-R1：五种 MEC 导航专家训练进度", "",
             f"更新时间：{__import__('time').strftime('%Y-%m-%d %H:%M:%S')}", "",
             "Dev = 训练池 2% episode 独立留出（与 Val-Seen 2498 任务无重叠）；"
             "Best 由 Dev Macro-F1 选择，不使用 Val-Seen。", "",
             "| 模型 | 目标FLOPs | 参数量 | Best Epoch | Train Time | "
             "Dev Macro-F1 | Val-Seen SR | OSR | SPL | NE | nDTW |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in states:
        sm = s["summary"]
        lines.append(
            f"| {s['name']} | {s['target']} G | {s['params']/1e6:.1f} M | "
            f"{'Pending' if s['best_epoch'] is None else s['best_epoch'] + 1} | "
            f"{fmt(s['train_h'], 2, ' h')} | {fmt(s['best_f1'])} | "
            f"{fmt(sm['sr'] if sm else None)} | "
            f"{fmt(sm['osr'] if sm else None)} | "
            f"{fmt(sm['spl'] if sm else None)} | "
            f"{fmt(sm['ne_m'] if sm else None, 2)} | "
            f"{fmt(sm['ndtw'] if sm else None)} |")
    lines += ["", "## 实时状态", ""]
    if cur:
        p = cur["progress"] or {}
        accum = int((cur.get("config") or {}).get("accum", 1))
        ep = p.get("epoch")
        upd = p.get("optimizer_step")
        micro = p.get("micro_step")
        in_epoch = False
        if live and (ep is None or live["epoch"] > ep):
            prev = ep
            ep = live["epoch"]
            base = p.get("optimizer_step") or 0
            upd = (base if prev == live["epoch"] - 1 else 0) + live["update"]
            micro = live["update"] * accum
            in_epoch = True
        loss = p.get("train_loss")
        if loss is None and live:
            loss = live["loss"]
        gpu = p.get("gpu") or live_gpu()
        ep_time = p.get("epoch_seconds")
        if in_epoch and live:
            ep_time = live["elapsed"]
        elif ep_time is None:
            ep_time = epoch_est_s
        lines += [
            "```text",
            f"当前模型：{cur['name']}",
            f"当前Epoch / 20：{'—' if ep is None else ep + 1} / 20",
            f"当前optimizer step：{num(upd, 0)}"
            + (f"（Epoch 内 {live['update']}/{live['updates']}）"
               if live and in_epoch else ""),
            f"当前micro-step：{num(micro, 0)}",
            f"Train Loss：{num(loss)}",
            f"Dev Macro-F1：{num(p.get('dev_macro_f1'))}",
            f"Learning Rate：{num(p.get('lr'), 7)}",
            f"Gradient Norm：{num(p.get('grad_norm'))}",
            f"GPU显存：{gpu.get('mem_mib', '?')} MiB "
            f"(峰值 {num(p.get('peak_mem_gb'), 2)} GB)",
            f"GPU利用率：{gpu.get('gpu_util_pct', '?')} %",
            f"当前Epoch耗时：{num((ep_time or 0)/60, 1)} min"
            + ("（进行中）" if in_epoch or p.get("epoch_seconds") is None else ""),
            f"当前模型预计剩余时间："
            f"{'—' if eta_cur is None else f'{eta_cur:.1f} h'}"
            f"（按 {EXPECTED_EPOCHS} epochs 早停口径）",
            f"剩余模型预计训练时间：{remain_h:.1f} h",
            f"预计Val-Seen评测时间：{EVAL_H:.1f} h/模型",
            "```", ""]
    else:
        lines += ["当前无训练进程（或训练队列未启动）。", ""]
    lines += ["## 剩余时间估计（基于实测 Epoch 耗时）", "",
              f"剩余训练合计约 **{remain_h:.1f} h**；明细：{estimate_text or '无'}。",
              f"闭环评测按 {EVAL_H:.1f} h/模型预留。", "",
              "## 图", "",
              "- `figures/sr_by_model.png`（已完成的 Val-Seen SR）",
              "- `figures/loss_curves.png`（Train/Dev Loss）",
              "- `figures/dev_macro_f1.png`（Dev Macro-F1）",
              "- `figures/compute_vs_params.png`（计算量 vs 参数量）",
              "- `MODEL_COMPARISON.md`（跨模型同 epoch 对比、闭环指标对比）", "",
              "未完成模型保持 Pending，未填入任何估计 SR。", ""]
    text = "\n".join(lines)
    (HERE / "PROGRESS_REPORT.md").write_text(text, encoding="utf-8")
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    (HERE / "results/PROGRESS_REPORT.md").write_text(text, encoding="utf-8")
    print(f"wrote PROGRESS_REPORT.md (remaining training ~{remain_h:.1f} h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
