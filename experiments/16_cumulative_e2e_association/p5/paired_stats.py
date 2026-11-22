#!/usr/bin/env python3
"""Stage 16-P5: paired per-task statistics across all eight policies."""
import json, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest

HERE=Path(__file__).resolve().parent
EXP16=HERE.parent
RNG=np.random.default_rng(42); NBOOT=10000
FILES={
 "random":EXP16/"p2/results/results_random.csv",
 "min_latency":EXP16/"p2/results/results_min_latency.csv",
 "max_quality":EXP16/"p2/results/results_max_quality.csv",
 "dqn":EXP16/"p4/runs/dqn/seed42/eval_corrected/episode_results.csv",
 "td3":EXP16/"p4/runs/td3/seed42/eval/episode_results.csv",
 "sac":EXP16/"p4/runs/sac/seed42/eval/episode_results.csv",
 "ppo":EXP16/"p4/runs/ppo/seed42/eval/episode_results.csv",
 "ddpg":EXP16/"p4/runs/ddpg/seed42/eval/episode_results.csv",
}
def load():
    out={}
    for k,p in FILES.items():
        if not p.exists(): print("missing",k,p); continue
        d=pd.read_csv(p).set_index("episode_id")
        d["success"]=d["success"].astype(str).str.lower().isin(["true","1"])
        out[k]=d
    ids=sorted(set.intersection(*[set(d.index) for d in out.values()]))
    return {k:d.loc[ids] for k,d in out.items()}, ids
def main():
    D,ids=load(); n=len(ids); print("aligned tasks:",n)
    rows=[]
    for a,b in itertools.combinations(D,2):
        sa=D[a]["success"].to_numpy(); sb=D[b]["success"].to_numpy()
        ca=D[a]["cum_e2e_ms"].to_numpy(); cb=D[b]["cum_e2e_ms"].to_numpy()
        a_only=int((sa&~sb).sum()); b_only=int((~sa&sb).sum())
        p=float(binomtest(min(a_only,b_only),a_only+b_only,0.5).pvalue) if a_only+b_only else 1.0
        diffs=np.empty(NBOOT); idx=RNG.integers(0,n,size=(NBOOT,n))
        # chunk to limit memory
        for i in range(0,NBOOT,1000):
            j=idx[i:i+1000]
            diffs[i:i+1000]=(ca[j]-cb[j]).mean(axis=1)
        lo,hi=np.percentile(diffs,[2.5,97.5])
        rows.append({"a":a,"b":b,"sr_a":float(sa.mean()),"sr_b":float(sb.mean()),
                     "sr_diff_a_minus_b":float(sa.mean()-sb.mean()),
                     "a_only":a_only,"b_only":b_only,"mcnemar_p":p,
                     "cum_e2e_a":float(ca.mean()),"cum_e2e_b":float(cb.mean()),
                     "cum_e2e_diff_a_minus_b":float(ca.mean()-cb.mean()),
                     "cum_e2e_ci_low":float(lo),"cum_e2e_ci_high":float(hi)})
    df=pd.DataFrame(rows); df.to_csv(HERE/"paired_stats.csv",index=False)
    # key comparisons vs each DRL
    key=[]
    for drl in ["dqn","td3","sac","ppo","ddpg"]:
        for base in ["random","min_latency","max_quality"]:
            r=df[((df.a==drl)&(df.b==base))|((df.a==base)&(df.b==drl))].iloc[0]
            sign=1 if r.a==drl else -1
            key.append({"drl":drl,"baseline":base,
                        "sr_diff":sign*r.sr_diff_a_minus_b,"mcnemar_p":r.mcnemar_p,
                        "cum_e2e_diff":sign*r.cum_e2e_diff_a_minus_b,
                        "ci_low":min(sign*r.cum_e2e_ci_low,sign*r.cum_e2e_ci_high),
                        "ci_high":max(sign*r.cum_e2e_ci_low,sign*r.cum_e2e_ci_high)})
    pd.DataFrame(key).to_csv(HERE/"key_comparisons.csv",index=False)
    print(pd.DataFrame(key).to_string(index=False))
if __name__=="__main__": main()
