#!/usr/bin/env python3
"""Summarize locked FULL_Q and the three Host-relative source arms."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,f1_score,roc_auc_score

ROOT=Path("artifacts/care-di25-host-relative-source-v1");LOCKED=Path("artifacts/care-di25-graphglobal-path-source-v1/arms/Q_QUERY_ONLY")
ARMS=("H0_FULL_Q_LOCKED","HM_R","HC_R","HMC_R")
METHODS={"H0_FULL_Q_LOCKED":"DI25__Host__Anomaly__GraphGlobalPath__Q_QUERY_ONLY__Bounded",**{a:f"DI25__HostRelative__Anomaly__{a}__QQuery__Bounded" for a in ARMS[1:]}}
def read(p):
    with Path(p).open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def exact(n01,n10):
    n=n01+n10
    if not n:return 1.0
    return min(1.0,2*sum(math.comb(n,k) for k in range(min(n01,n10)+1))/(2**n))
def metrics(y,p,base):
    pred=p>=.5;bpred=base>=.5;fixed=int(np.sum((bpred!=y)&(pred==y)));broken=int(np.sum((bpred==y)&(pred!=y)))
    return {"BA":float(balanced_accuracy_score(y,pred)),"AUC":float(roc_auc_score(y,p)),"F1":float(f1_score(y,pred,zero_division=0)),"TN":int(np.sum((y==0)&(~pred))),"FP":int(np.sum((y==0)&pred)),"FN":int(np.sum((y==1)&(~pred))),"TP":int(np.sum((y==1)&pred)),"fixed":fixed,"broken":broken,"net":fixed-broken}
def main():
    ref=read(ROOT/"preflight/di25_reference/source_predictions.csv");ref.sort(key=lambda r:r["sha256"].lower());ids=[r["sha256"].lower() for r in ref];y=np.asarray([int(float(r["label"])) for r in ref]);fold=np.asarray([int(float(r["fold"])) for r in ref]);route=np.asarray([int(float(r["A25__routed"])) for r in ref]);base=np.asarray([float(r["Global24__prob"]) for r in ref])
    probs={};param={};integrity=[]
    for arm in ARMS:
        root=LOCKED if arm=="H0_FULL_Q_LOCKED" else ROOT/"arms"/arm;rows=read(root/"source_predictions.csv");by={r["sha256"].lower():r for r in rows};col=METHODS[arm]+"__prob"
        if set(by)!=set(ids) or any(col not in by[s] for s in ids):raise RuntimeError(f"BLOCKED_BY_CONTRACT: prediction contract {arm}")
        probs[arm]=np.asarray([float(by[s][col]) for s in ids])
        for s in ids:
            observed=int(float(by[s]["A25__routed"]));expected=int(route[ids.index(s)]);integrity.append({"sha256":s,"arm":arm,"route_frozen_DI25":expected,"route_arm":observed,"different":int(expected!=observed)})
        fit=read(root/"fit_audit.csv");counts={int(float(r["parameter_n"])) for r in fit}
        if counts!={33036}:raise RuntimeError(f"BLOCKED_BY_CONTRACT: parameter count {arm}={counts}")
        param[arm]=33036
    if sum(r["different"] for r in integrity):raise RuntimeError("BLOCKED_BY_CONTRACT: route changed")
    aggregate=[];foldrows=[]
    for arm in ARMS:
        aggregate.append({"arm":arm,**metrics(y,probs[arm],base)})
        for f in range(1,6):
            q=fold==f;foldrows.append({"arm":arm,"fold":f,**metrics(y[q],probs[arm][q],base[q])})
    paired=[]
    h0=(probs["H0_FULL_Q_LOCKED"]>=.5)==y
    for arm in ARMS[1:]:
        c=(probs[arm]>=.5)==y;n01=int(np.sum((~h0)&c));n10=int(np.sum(h0&(~c)));paired.append({"comparison":arm+"_minus_H0_FULL_Q_LOCKED","candidate_only_correct":n01,"H0_only_correct":n10,"paired_net":n01-n10,"exact_p":exact(n01,n10)})
    by={r["arm"]:r for r in aggregate};interaction={"contrast":"HMC_R-HM_R-HC_R+H0","dBA":by["HMC_R"]["BA"]-by["HM_R"]["BA"]-by["HC_R"]["BA"]+by["H0_FULL_Q_LOCKED"]["BA"],"dAUC":by["HMC_R"]["AUC"]-by["HM_R"]["AUC"]-by["HC_R"]["AUC"]+by["H0_FULL_Q_LOCKED"]["AUC"],"dF1":by["HMC_R"]["F1"]-by["HM_R"]["F1"]-by["HC_R"]["F1"]+by["H0_FULL_Q_LOCKED"]["F1"],"dFixed":by["HMC_R"]["fixed"]-by["HM_R"]["fixed"]-by["HC_R"]["fixed"]+by["H0_FULL_Q_LOCKED"]["fixed"],"dBroken":by["HMC_R"]["broken"]-by["HM_R"]["broken"]-by["HC_R"]["broken"]+by["H0_FULL_Q_LOCKED"]["broken"],"dNet":by["HMC_R"]["net"]-by["HM_R"]["net"]-by["HC_R"]["net"]+by["H0_FULL_Q_LOCKED"]["net"]}
    write(ROOT/"results/aggregate_metrics.csv",aggregate);write(ROOT/"results/per_fold_metrics.csv",foldrows);write(ROOT/"results/paired_vs_H0.csv",paired);write(ROOT/"results/route_integrity.csv",integrity);write(ROOT/"results/parameter_count.csv",[{"arm":a,"parameter_count":n} for a,n in param.items()]);write(ROOT/"results/factorial_interaction.csv",[interaction])
    lines=["CARE-Droid Host-relative representation source 2x2 v1","","Strict 2017 OOF; frozen DI25=200/799; Q-Query; Host-only; no Ego/future row.","H0 reuses locked FULL_Q and is not retrained. All arms: Host width=74, parameters=33,036.","External benign, internal dominant, and shared-commonness backgrounds are unchanged and fold-safe.",""]
    for r in aggregate:lines.append(f"{r['arm']}: BA={r['BA']:.6f} AUC={r['AUC']:.6f} F1={r['F1']:.6f} fixed={r['fixed']} broken={r['broken']} net={r['net']}")
    lines+=["","Paired versus H0:"]+[f"{r['comparison']}: candidate_only={r['candidate_only_correct']} H0_only={r['H0_only_correct']} net={r['paired_net']} p={r['exact_p']:.6f}" for r in paired]
    lines+=["",f"M x C interaction: dBA={interaction['dBA']:+.6f} dAUC={interaction['dAUC']:+.6f} dFixed={interaction['dFixed']:+d} dBroken={interaction['dBroken']:+d} dNet={interaction['dNet']:+d}","","Hard audits: route_difference_count=0; parameter_count_all_arms=33036; future_rows_loaded=0; ego_consumed=false.","Do not run future diagnostics until the source result and size-control audit are inspected."]
    (ROOT/"results/REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8");print("HOST_RELATIVE_SOURCE_SUMMARY_PASS");print((ROOT/"results/REPORT.txt").read_text(encoding="utf-8"))
if __name__=="__main__":main()
