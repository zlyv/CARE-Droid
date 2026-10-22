#!/usr/bin/env python3
from __future__ import annotations
import csv,math
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score,confusion_matrix
ROOT=Path("artifacts/care-di25-roleconditioned-background-source-v1");H0=Path("artifacts/care-di25-graphglobal-path-source-v1/arms/Q_QUERY_ONLY")
ARMS=("B0_H0_LOCKED","B1_HARD_ROLE","B2_SOFT_ROLE");METHOD={"B0_H0_LOCKED":"DI25__Host__Anomaly__GraphGlobalPath__Q_QUERY_ONLY__Bounded","B1_HARD_ROLE":"DI25__Host__Anomaly__QQuery__HardRoleBackground__Bounded","B2_SOFT_ROLE":"DI25__Host__Anomaly__QQuery__SoftRoleBackground__Bounded"}
def read(p):
    with Path(p).open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def exact(a,b):
    n=a+b
    return 1.0 if not n else min(1.0,2*sum(math.comb(n,k) for k in range(min(a,b)+1))/(2**n))
def metric(y,p,base):
    pred=p>=.5;b=base>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();fixed=int(np.sum((b!=y)&(pred==y)));broken=int(np.sum((b==y)&(pred!=y)))
    return {"BA":float(balanced_accuracy_score(y,pred)),"AUC":float(roc_auc_score(y,p)),"F1":float(f1_score(y,pred)),"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),"fixed":fixed,"broken":broken,"net":fixed-broken}
def main():
    ref=read(ROOT/"preflight/di25_reference/source_predictions.csv");ref.sort(key=lambda r:r["sha256"].lower());ids=[r["sha256"].lower() for r in ref];y=np.asarray([int(float(r["label"])) for r in ref]);fold=np.asarray([int(float(r["fold"])) for r in ref]);route=np.asarray([int(float(r["A25__routed"])) for r in ref]);base=np.asarray([float(r["Global24__prob"]) for r in ref])
    probs={};integrity=[]
    for arm in ARMS:
        root=H0 if arm=="B0_H0_LOCKED" else ROOT/"arms"/arm;rows=read(root/"source_predictions.csv");by={r["sha256"].lower():r for r in rows};col=METHOD[arm]+"__prob"
        if set(by)!=set(ids):raise RuntimeError("BLOCKED_BY_CONTRACT: prediction cohort "+arm)
        probs[arm]=np.asarray([float(by[s][col]) for s in ids])
        for i,s in enumerate(ids):integrity.append({"sha256":s,"arm":arm,"route_DI":int(route[i]),"route_arm":int(float(by[s]["A25__routed"])),"different":int(int(route[i])!=int(float(by[s]["A25__routed"])))})
    if sum(x["different"] for x in integrity):raise RuntimeError("BLOCKED_BY_CONTRACT: route changed")
    aggregate=[{"arm":a,**metric(y,probs[a],base)} for a in ARMS];foldrows=[{"arm":a,"fold":f,**metric(y[fold==f],probs[a][fold==f],base[fold==f])} for a in ARMS for f in range(1,6)]
    h0=(probs[ARMS[0]]>=.5)==y;paired=[]
    for a in ARMS[1:]:
        c=(probs[a]>=.5)==y;n01=int(np.sum(~h0&c));n10=int(np.sum(h0&~c));paired.append({"comparison":a+"_minus_B0_H0_LOCKED","candidate_only_correct":n01,"H0_only_correct":n10,"paired_net":n01-n10,"exact_p":exact(n01,n10)})
    write(ROOT/"results/aggregate_metrics.csv",aggregate);write(ROOT/"results/per_fold_metrics.csv",foldrows);write(ROOT/"results/paired_vs_H0.csv",paired);write(ROOT/"results/route_integrity.csv",integrity)
    lines=["CARE-Droid capacity-matched role-conditioned three-background source v1","","Strict 2017 five-fold OOF; frozen DI25=200/799; original H0 74-D Host; Q-Query; no Ego/future row.","All arms retain shared=32, benign=24 prototypes and 33,036 neural parameters.","The 31-D endpoint-role sidecar selects backgrounds only and never enters the neural correction vector.",""]
    for r in aggregate:lines.append(f"{r['arm']}: BA={r['BA']:.6f} AUC={r['AUC']:.6f} F1={r['F1']:.6f} fixed={r['fixed']} broken={r['broken']} net={r['net']}")
    lines+=["","Paired versus H0:"]+[f"{r['comparison']}: candidate_only={r['candidate_only_correct']} H0_only={r['H0_only_correct']} net={r['paired_net']} p={r['exact_p']:.6f}" for r in paired]
    lines+=["","Do not run future diagnostics until one source candidate is explicitly frozen."]
    (ROOT/"results/REPORT.txt").write_text("\n".join(lines)+"\n");print("ROLECONDITIONED_BACKGROUND_SOURCE_SUMMARY_PASS");print((ROOT/"results/REPORT.txt").read_text())
if __name__=="__main__":main()
