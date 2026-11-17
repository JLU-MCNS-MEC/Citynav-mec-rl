#!/usr/bin/env python3
"""Stage 14-R1 live report: PROGRESS_REPORT.md + figures.

Reads results/<Model>/ (training history, closed-loop summary) and writes
  PROGRESS_REPORT.md (also results/PROGRESS_REPORT.md)
  figures/{sr_by_model,loss_curves,dev_macro_f1,compute_vs_params}.png
Never fabricates numbers: unfinished models stay Pending.
"""
from __future__ import annotations

import json
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
    plt.legend(fontsize=7, ncol=2); plt.title("Training / Dev loss")
    plt.tight_layout(); plt.savefig(figdir / "loss_curves.png", dpi=130); plt.close()
    # Dev Macro-F1
    plt.figure(figsize=(8, 4.5))
    for s in states:
        if s["hist"]:
            ep = [h["epoch"] + 1 for h in s["hist"]]
            plt.plot(ep, [h["dev"]["macro_f1"] for h in s["hist"]], marker="o",
                     ms=3, label=s["name"])
    plt.xlabel("epoch"); plt.ylabel("Dev Macro-F1"); plt.grid(alpha=.3)
    plt.legend(fontsize=8); plt.title("Independent Dev Macro-F1 (2% hold-out)")
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


def main() -> int:
    states = [model_state(k) for k in KEYS]
    figures(states)
    remain_h, estimate_text = estimates(states)
    cur = None
    for s in states:
        if s["progress"] and not s["progress"].get("done"):
            cur = s
            break
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
            f"{'Pending' if s['best_epoch'] is None else s['best_epoch']} | "
            f"{fmt(s['train_h'], 2, ' h')} | {fmt(s['best_f1'])} | "
            f"{fmt(sm['sr'] if sm else None)} | "
            f"{fmt(sm['osr'] if sm else None)} | "
            f"{fmt(sm['spl'] if sm else None)} | "
            f"{fmt(sm['ne_m'] if sm else None, 2)} | "
            f"{fmt(sm['ndtw'] if sm else None)} |")
    lines += ["", "## 实时状态", ""]
    if cur:
        p = cur["progress"]
        eta = None
        if cur["epoch_sec"]:
            left = max(0, EXPECTED_EPOCHS - (p.get("epoch", 0) + 1))
            eta = left * cur["epoch_sec"] / 3600
        lines += [
            "```text",
            f"当前模型：{cur['name']}",
            f"当前Epoch / 20：{p.get('epoch', 0) + 1} / {p.get('epochs', 20)}",
            f"当前optimizer step：{p.get('optimizer_step')}",
            f"当前micro-step：{p.get('micro_step')}",
            f"Train Loss：{num(p.get('train_loss'))}",
            f"Dev Macro-F1：{num(p.get('dev_macro_f1'))}",
            f"Learning Rate：{num(p.get('lr'), 7)}",
            f"Gradient Norm：{num(p.get('grad_norm'))}",
            f"GPU显存：{p.get('gpu', {}).get('mem_mib', '?')} MiB "
            f"(峰值 {num(p.get('peak_mem_gb'), 2)} GB)",
            f"GPU利用率：{p.get('gpu', {}).get('gpu_util_pct', '?')} %",
            f"当前Epoch耗时：{num((p.get('epoch_seconds') or 0)/60, 1)} min",
            f"当前模型预计剩余时间："
            f"{'—' if eta is None else f'{eta:.1f} h'}"
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
              "- `figures/compute_vs_params.png`（计算量 vs 参数量）", "",
              "未完成模型保持 Pending，未填入任何估计 SR。", ""]
    text = "\n".join(lines)
    (HERE / "PROGRESS_REPORT.md").write_text(text, encoding="utf-8")
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    (HERE / "results/PROGRESS_REPORT.md").write_text(text, encoding="utf-8")
    print(f"wrote PROGRESS_REPORT.md (remaining training ~{remain_h:.1f} h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
