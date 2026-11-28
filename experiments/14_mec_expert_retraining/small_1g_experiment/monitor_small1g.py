#!/usr/bin/env python3
"""Small-1G per-epoch monitor: table vs Small-5G, ETA, figures."""
from __future__ import annotations
import json, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent; EXP14=HERE.parent
NEW=HERE/"runs/small1g/seed42/history.jsonl"
OLD=EXP14/"results/Small/training_history.jsonl"
FIG=HERE/"figures"

def f1_new(r): return float(r.get("dev_macro_f1", float("nan")))
def f1_old(r): return float(r["dev"].get("macro_f1", float("nan")))

def load(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

def main():
    new=load(NEW); old={int(r["epoch"]):r for r in load(OLD)}
    newby={int(r["epoch"]):r for r in new}
    rows=[]
    for e in range(20):
        o=f1_old(old[e]) if e in old else None
        if e in newby:
            r=newby[e]; n=f1_new(r); loss=r.get("train_loss"); lr=r.get("lr"); sec=r.get("sec")
            st="本次训练"
        else:
            n=loss=lr=sec=None; st="Pending"
        rows.append((e,o,n,loss,lr,sec,st))
    best_ep=max(newby,key=lambda e:f1_new(newby[e])) if newby else None
    best=max((f1_new(r) for r in new),default=None)
    durs=[float(r.get("sec",0)) for r in new if r.get("sec")][-3:]
    rem=max(0,19-max(newby)) if newby else 20
    eta=(np.mean(durs)/60*rem) if durs else None
    L=["# Small-1G · 训练进度","",f"更新：{time.strftime('%Y-%m-%d %H:%M:%S')}","",
       f"- 当前 epoch：{max(newby) if newby else '-'} / 19",
       f"- 最佳 Dev Macro-F1：**{best:.4f}**（Epoch {best_ep}）" if best else "- 最佳：—",
       "- Small-5G 历史最佳：0.5771（Dev F1）",
       f"- 最近 3 epoch 平均：{np.mean(durs)/60:.1f} min；预计剩余 {eta:.0f} min" if eta else "",
       "", "| Epoch | Small-5G F1 | **Small-1G F1** | Train Loss | LR | 耗时 | 状态 |",
       "|---:|---:|---:|---:|---:|---:|---|"]
    fm=lambda v:"—" if v is None else f"{v:.4f}"
    for e,o,n,loss,lr,sec,st in rows:
        L.append(f"| {e} | {fm(o)} | {fm(n)} | {'—' if loss is None else f'{loss:.4f}'} | "
                 f"{'—' if lr is None else f'{lr:.2e}'} | {'—' if sec is None else f'{sec/60:.1f} min'} | {st} |")
    (HERE/"PROGRESS_REPORT.md").write_text("\n".join(L)+"\n")
    (HERE/"progress.json").write_text(json.dumps({"epoch":max(newby) if newby else None,
        "best_f1":best,"best_epoch":best_ep,"remaining":rem,"eta_min":eta},indent=2))
    FIG.mkdir(exist_ok=True)
    if newby:
        ep=sorted(newby)
        fig,ax=plt.subplots(figsize=(8,4.4),dpi=140)
        ax.plot(sorted(old),[f1_old(old[k]) for k in sorted(old)],"s-",label="Small-5G",color="#1f77b4")
        ax.plot(ep,[f1_new(newby[e]) for e in ep],"^-",label="Small-1G",color="#d62728")
        ax.set_xlabel("epoch");ax.set_ylabel("Dev Macro-F1");ax.grid(alpha=.3);ax.legend(frameon=False)
        fig.tight_layout();fig.savefig(FIG/"dev_f1.png");plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4.4),dpi=140)
        ax.plot(sorted(old),[old[k]["train_loss"] for k in sorted(old)],"s-",label="Small-5G",color="#1f77b4")
        ax.plot(ep,[newby[e]["train_loss"] for e in ep],"^-",label="Small-1G",color="#d62728")
        ax.set_xlabel("epoch");ax.set_ylabel("train loss");ax.grid(alpha=.3);ax.legend(frameon=False)
        fig.tight_layout();fig.savefig(FIG/"train_loss.png");plt.close(fig)
    print("\n".join(L[5:]))

if __name__=="__main__": main()
