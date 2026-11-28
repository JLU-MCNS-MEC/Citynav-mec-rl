#!/usr/bin/env python3
"""Stage 14-R4 monitor: 4-model Dev F1 table (old Large / TS / TM-R3 / Large-R4)."""
from __future__ import annotations
import json, time
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent; EXP14=HERE.parent
NEW=HERE/"runs/large/seed42/history.jsonl"
HIST={"Large(orig)":EXP14/"stability_revision/runs/large/seed42/training_history.json",
      "TS":EXP14/"stability_revision/runs/ts/seed42/training_history.json",
      "TM-R3":EXP14/"tm_r3_lr_continuation/runs/tm/seed42/training_history.jsonl"}
FIG=HERE/"figures"

def f1(r):
    if "dev_macro_f1" in r: return float(r["dev_macro_f1"])
    return float(r.get("dev",{}).get("macro_f1", float("nan")))
def load(p):
    if not p.exists(): return []
    t=p.read_text()
    try:
        d=json.loads(t)
        return d if isinstance(d,list) else []
    except json.JSONDecodeError:
        return [json.loads(l) for l in t.splitlines() if l.strip()]

def main():
    new=load(NEW); newby={int(r["epoch"]):r for r in new}
    hist={k:{int(r["epoch"]):r for r in load(p)} for k,p in HIST.items()}
    rows=[]
    for e in range(20):
        vals={k:(f1(v[e]) if e in v else None) for k,v in hist.items()}
        n=f1(newby[e]) if e in newby else None
        rows.append((e,vals.get("Large(orig)"),vals.get("TS"),vals.get("TM-R3"),n))
    best=max(((f1(r),int(r["epoch"])) for r in new),default=(None,None))
    durs=[float(r.get("seconds",r.get("sec",0))) for r in new][-3:]
    rem=max(0,19-max(newby)) if newby else 18
    eta=np.mean(durs)/60*rem if durs else None
    L=["# Stage 14-R4 · Large 低学习率续训进度","",
       f"更新：{time.strftime('%Y-%m-%d %H:%M:%S')}","",
       f"- 当前 epoch：{max(newby) if newby else '-'}/19",
       f"- Large-R4 最佳：**{best[0]:.4f}**（Epoch {best[1]}）" if best[0] else "- 最佳：—",
       "- 历史基准：原 Large 0.6105（ep1）/ TM-R3 0.6407（ep15）/ TS 0.6500（ep16）",
       f"- 最近 3 epoch 均值 {np.mean(durs)/60:.1f} min；预计剩余 {eta:.0f} min" if eta else "",
       "", "| Ep | 原 Large | TS | TM-R3 | **Large-R4** | Train Loss | LR | 耗时 |",
       "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fm=lambda v:"—" if v is None else f"{v:.4f}"
    for e,o,t,m,n in rows:
        r=newby.get(e); loss=r.get("train_loss") if r else None
        lr=r.get("lr") if r else None; sec=r.get("seconds", r.get("sec")) if r else None
        L.append(f"| {e} | {fm(o)} | {fm(t)} | {fm(m)} | {fm(n)} | "
                 f"{'—' if loss is None else f'{loss:.4f}'} | {'—' if lr is None else f'{lr:.2e}'} | "
                 f"{'—' if sec is None else f'{sec/60:.1f} min'} |")
    (HERE/"PROGRESS_REPORT.md").write_text("\n".join(L)+"\n")
    (HERE/"progress.json").write_text(json.dumps({"epoch":max(newby) if newby else None,
        "best_f1":best[0],"best_epoch":best[1],"eta_min":eta,"remaining":rem},indent=2))
    FIG.mkdir(exist_ok=True)
    if newby:
        ep=sorted(newby)
        fig,ax=plt.subplots(figsize=(8.4,4.6),dpi=140)
        for k,v in hist.items():
            ax.plot(sorted(v),[f1(v[e]) for e in sorted(v)],"s-",ms=3,label=k,alpha=.7)
        ax.plot(ep,[f1(newby[e]) for e in ep],"^-",label="Large-R4",color="#d62728")
        ax.set_xlabel("epoch");ax.set_ylabel("Dev Macro-F1");ax.grid(alpha=.3);ax.legend(frameon=False)
        fig.tight_layout();fig.savefig(FIG/"dev_f1_4models.png");plt.close(fig)
    print("\n".join(L[5:]))

if __name__=="__main__": main()
