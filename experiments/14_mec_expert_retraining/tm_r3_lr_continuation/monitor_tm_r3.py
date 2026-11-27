#!/usr/bin/env python3
"""Stage 14-R3 monitor: 20-epoch TS/TM/new-TM table, ETA, figures.

Safe to run repeatedly; it only reads training outputs and rewrites
PROGRESS_REPORT.md / progress.json / figures.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EXP14 = HERE.parent

TS_HIST = EXP14 / "stability_revision/runs/ts/seed42/training_history.json"
TM_HIST = EXP14 / "stability_revision/runs/tm/seed42/training_history.json"
NEW_DIR = HERE / "runs/tm/seed42"
NEW_HIST = NEW_DIR / "training_history.jsonl"
FIG = HERE / "figures"
TOTAL_EPOCHS = 20


def load_json_list(p: Path):
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    return d if isinstance(d, list) else []


def load_jsonl(p: Path):
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def f1_of(h):
    return float(h["dev"]["macro_f1"])


def main():
    ts = {int(h["epoch"]): h for h in load_json_list(TS_HIST)}
    tm = {int(h["epoch"]): h for h in load_json_list(TM_HIST)}
    new = load_jsonl(NEW_HIST)
    new_by = {int(h["epoch"]): h for h in new}

    rows = []
    for e in range(TOTAL_EPOCHS):
        ts_f1 = f1_of(ts[e]) if e in ts else None
        tm_f1 = f1_of(tm[e]) if e in tm else None
        if e in new_by:
            nf1 = f1_of(new_by[e])
            loss = new_by[e].get("train_loss")
            lr = new_by[e].get("lr")
            status = "本次训练"
        elif e in (0, 1):
            nf1, loss, lr = tm_f1, None, None
            status = "历史继承"
        else:
            nf1, loss, lr, status = None, None, None, "Pending"
        rows.append((e, ts_f1, tm_f1, nf1, loss, lr, status))

    # ETA from the last three completed epochs
    durations = [float(h.get("seconds", h.get("epoch_seconds", 0.0)))
                 for h in new if h.get("seconds") or h.get("epoch_seconds")][-3:]
    remaining_epochs = max(0, TOTAL_EPOCHS - 1 - max(new_by) ) if new_by else 18
    eta_min = (np.mean(durations) / 60.0) * remaining_epochs if durations else None
    finish = (time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + eta_min * 60))
              if eta_min else "未知")

    best_epoch = max(new_by, key=lambda e: f1_of(new_by[e])) if new_by else None
    best_f1 = f1_of(new_by[best_epoch]) if new_by else None

    lines = ["# Stage 14-R3 · Transformer-M 低学习率续训进度", "",
             f"更新时间：{time.strftime('%Y-%m-%d %H:%M:%S')}", "",
             f"- 当前 Epoch：**{max(new_by) if new_by else '-'} / 19**",
             f"- 新 TM 最佳 Dev Macro-F1：**{best_f1:.4f}**（Epoch {best_epoch}）"
             if best_f1 is not None else "- 新 TM 最佳：尚未开始",
             "- 历史 TM 最佳：0.6177（Epoch 1）；历史 TS 最佳：0.6500（Epoch 16）",
             f"- 最近三个 Epoch 平均耗时：{np.mean(durations)/60:.1f} min"
             if durations else "- 尚无 Epoch 耗时",
             f"- 预计剩余：{eta_min:.0f} min；预计完成：{finish}" if eta_min else "",
             "", "| Epoch | 原 TS Dev F1 | 原 TM Dev F1 | 新 TM Dev F1 | 新 TM Train Loss | LR | 状态 |",
             "|---:|---:|---:|---:|---:|---:|---|"]
    for e, a, b, c, loss, lr, st in rows:
        fmt = lambda v: "—" if v is None else f"{v:.4f}"
        lrs = "—" if lr is None else f"{lr:.2e}"
        loss_s = "—" if loss is None else f"{loss:.4f}"
        lines.append(f"| {e} | {fmt(a)} | {fmt(b)} | {fmt(c)} | {loss_s} | {lrs} | {st} |")
    lines += ["", "> Epoch 0/1 的新 TM 数值为历史继承结果，不是本轮重新训练所得。",
              "> Pending 表示尚未训练，不做插值。", ""]
    (HERE / "PROGRESS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    progress = {"updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "current_epoch": max(new_by) if new_by else None,
                "best_epoch": best_epoch, "best_dev_f1": best_f1,
                "remaining_epochs": remaining_epochs,
                "eta_min": eta_min, "epochs_done": sorted(new_by)}
    (HERE / "progress.json").write_text(json.dumps(progress, indent=2))

    FIG.mkdir(exist_ok=True)
    if new_by:
        ep = sorted(new_by)
        fig, ax = plt.subplots(figsize=(8, 4.4), dpi=140)
        ax.plot([e for e in ts], [f1_of(ts[e]) for e in ts], "o-", label="TS (history)", color="#1f77b4")
        ax.plot([e for e in tm], [f1_of(tm[e]) for e in tm], "s-", label="TM (history)", color="#2ca02c")
        ax.plot(ep, [f1_of(new_by[e]) for e in ep], "^-", label="TM (R3)", color="#d62728")
        ax.axhline(0.6177, ls=":", color="gray", lw=1)
        ax.axhline(0.6500, ls=":", color="gray", lw=1)
        ax.set_xlabel("epoch"); ax.set_ylabel("Dev Macro-F1"); ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(FIG / "dev_f1.png"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.4), dpi=140)
        ax.plot(ep, [new_by[e]["train_loss"] for e in ep], "o-", color="#d62728")
        ax.set_xlabel("epoch"); ax.set_ylabel("train loss"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG / "train_loss.png"); plt.close(fig)

        lrs = [(e, new_by[e].get("lr")) for e in ep if new_by[e].get("lr")]
        if lrs:
            fig, ax = plt.subplots(figsize=(8, 4.4), dpi=140)
            ax.plot([a for a, _ in lrs], [b for _, b in lrs], "o-", color="#9467bd")
            ax.set_xlabel("epoch"); ax.set_ylabel("learning rate"); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(FIG / "lr.png"); plt.close(fig)

    print(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
