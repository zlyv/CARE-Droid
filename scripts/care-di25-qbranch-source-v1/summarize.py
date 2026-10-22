#!/usr/bin/env python3
"""Read-only paired summary of the strict-2017 Q-branch experiment."""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import binomtest

HERE=Path(__file__).resolve().parent; PROJECT=HERE.parent.parent
sys.path[:0]=[str(HERE),str(HERE.parent/"care-di25-graphglobal-path-source-v1")]
from qbranch_models import ARM_SPECS
ROOT=Path("artifacts/care-di25-qbranch-source-v1")
FULL=Path("artifacts/care-di25-graphglobal-path-source-v1/arms/Q_QUERY_ONLY")

def read(p):
    with Path(p).open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def write(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with p.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
def prob_col(rows):
    cs=[c for c in rows[0] if c.endswith("__Bounded__prob")]
    if len(cs)!=1:raise RuntimeError("Expected one bounded probability column, got "+repr(cs))
    return cs[0]
def metrics(y,p,base,route):
    pred=p>=.5;bpred=base>=.5
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    fixed=int(np.sum((pred==y)&(bpred!=y)));broken=int(np.sum((pred!=y)&(bpred==y)))
    return dict(BA=float(balanced_accuracy_score(y,pred)),AUC=float(roc_auc_score(y,p)),
      F1=float(f1_score(y,pred)),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),
      fixed=fixed,broken=broken,net=fixed-broken,routed=int(route.sum()))
def exact(a,b,y):
    ca=(a>=.5)==y;cb=(b>=.5)==y
    n01=int(np.sum((~ca)&cb));n10=int(np.sum(ca&(~cb)))
    return n01,n10,float(binomtest(min(n01,n10),n01+n10,.5).pvalue) if n01+n10 else 1.0

def main():
    paths={"FULL_Q":FULL/"source_predictions.csv"}
    for a in ARM_SPECS:
        if a!="FULL_Q":paths[a]=ROOT/"arms"/a/"source_predictions.csv"
    tables={a:read(p) for a,p in paths.items()}
    for a,r in tables.items():
        if len(r)!=799:raise RuntimeError(f"{a}: rows={len(r)}")
    ids=[r["sha256"].lower() for r in tables["FULL_Q"]]
    base=np.array([float(r["Global24__prob"]) for r in tables["FULL_Q"]]);y=np.array([int(float(r["label"])) for r in tables["FULL_Q"]])
    route=np.array([int(float(r["A25__routed"])) for r in tables["FULL_Q"]],dtype=bool)
    folds=np.array([int(float(r["fold"])) for r in tables["FULL_Q"]])
    if len(set(ids))!=799 or route.sum()!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: FULL_Q cohort/route changed")
    probs={};summary=[];fold_rows=[];route_diff=0;param={}
    for arm,rows in tables.items():
        by={r["sha256"].lower():r for r in rows}
        if set(by)!=set(ids):raise RuntimeError(f"{arm}: cohort mismatch")
        rr=[by[i] for i in ids];aroute=np.array([int(float(r["A25__routed"])) for r in rr],dtype=bool)
        route_diff+=int(np.sum(aroute!=route));probs[arm]=np.array([float(r[prob_col(rr)]) for r in rr])
        m=metrics(y,probs[arm],base,route);summary.append({"arm":arm,**m})
        for f in range(1,6):
            z=folds==f;fold_rows.append({"arm":arm,"fold":f,**metrics(y[z],probs[arm][z],base[z],route[z])})
        fit=read((FULL if arm=="FULL_Q" else ROOT/"arms"/arm)/"fit_audit.csv")
        pc={int(float(r["parameter_n"])) for r in fit}
        if pc!={33036}:raise RuntimeError(f"{arm}: parameter count {pc}")
        param[arm]=33036
    if route_diff:raise RuntimeError(f"BLOCKED_BY_CONTRACT: route difference count={route_diff}")
    locked={"BA":0.917359022556391,"AUC":0.9441917293233083,"F1":0.9147286821705426,
            "fixed":87,"broken":9,"net":78}
    observed=next(r for r in summary if r["arm"]=="FULL_Q")
    for k,v in locked.items():
        changed=(abs(float(observed[k])-v)>1e-12) if isinstance(v,float) else (int(observed[k])!=v)
        if changed:raise RuntimeError(f"BLOCKED_BY_CONTRACT: locked FULL_Q {k} changed: {observed[k]} != {v}")
    paired=[]
    for arm in ARM_SPECS:
        if arm=="FULL_Q":continue
        n01,n10,p=exact(probs["FULL_Q"],probs[arm],y)
        paired.append({"comparison":f"{arm}_minus_FULL_Q","candidate_only_correct":n01,"FULL_Q_only_correct":n10,"paired_net":n01-n10,"exact_p":p})
    # Direct Role-vs-Host complementarity on the frozen routed cohort.
    role=(probs["ROLE_ONLY"]>=.5)==y;host=(probs["HOST_ONLY_Q"]>=.5)==y;base_correct=(base>=.5)==y
    pair_detail=[]
    for i in range(799):
        if not route[i]:continue
        status=("both_correct" if role[i] and host[i] else "both_wrong" if not role[i] and not host[i]
                else "role_only_correct" if role[i] else "host_only_correct")
        pair_detail.append({"sha256":ids[i],"fold":int(folds[i]),"label":int(y[i]),"stage1_correct":int(base_correct[i]),
          "FULL_Q_prob":probs["FULL_Q"][i],"ROLE_ONLY_prob":probs["ROLE_ONLY"][i],"HOST_ONLY_Q_prob":probs["HOST_ONLY_Q"][i],
          "STRUCTURE_ONLY_Q_prob":probs["STRUCTURE_ONLY_Q"][i],"role_host_status":status})
    from collections import Counter
    c=Counter(r["role_host_status"] for r in pair_detail)
    err=route&(~base_correct);safe=route&base_correct
    complement={"routed_stage1_errors":int(err.sum()),"routed_stage1_correct":int(safe.sum()),
      "role_fixed_errors":int(np.sum(err&role)),"host_fixed_errors":int(np.sum(err&host)),
      "both_fixed_errors":int(np.sum(err&role&host)),"role_only_fixed_errors":int(np.sum(err&role&~host)),
      "host_only_fixed_errors":int(np.sum(err&~role&host)),"role_broken_safe":int(np.sum(safe&~role)),
      "host_broken_safe":int(np.sum(safe&~host)),**dict(c)}
    write(ROOT/"results/aggregate_metrics.csv",summary);write(ROOT/"results/fold_metrics.csv",fold_rows)
    write(ROOT/"results/paired_vs_FULL_Q.csv",paired);write(ROOT/"results/role_host_paired.csv",pair_detail)
    write(ROOT/"results/parameter_count.csv",[{"arm":a,"parameter_count":n,"active_dimensions":int(__import__('qbranch_models').correction_mask(a).sum())} for a,n in param.items()])
    lines=["CARE-Droid Q-Query correction-branch isolation source v1","",
      "Strict 2017 five-fold OOF; frozen DI25=200/799; Host-only; no Ego or future row.",
      "FULL_Q reuses the locked Q_QUERY_ONLY predictions and is not retrained.",
      "All trained arms retain the same 33,036 parameters and 113-D head; fixed zero masks isolate information branches.",""]
    for r in summary:lines.append(f"{r['arm']}: BA={r['BA']:.6f} AUC={r['AUC']:.6f} F1={r['F1']:.6f} fixed={r['fixed']} broken={r['broken']} net={r['net']}")
    lines += ["","Role vs Host on routed APKs:",*(f"{k}={v}" for k,v in complement.items()),"",
      "Interpretation boundary: sufficiency under retraining is not causal attribution. Freeze no future experiment until the source result is inspected.",
      "route_difference_count=0","parameter_count_all_arms=33036","future_rows_loaded=0","ego_consumed=false"]
    (ROOT/"results/REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (ROOT/"results/summary.json").write_text(json.dumps({"metrics":summary,"paired":paired,"complementarity":complement,"route_difference_count":0,"future_rows_loaded":0},indent=2)+"\n",encoding="utf-8")
    print("QBRANCH_SOURCE_SUMMARY_PASS");print((ROOT/"results/REPORT.txt").read_text(encoding="utf-8"))
if __name__=="__main__":main()
