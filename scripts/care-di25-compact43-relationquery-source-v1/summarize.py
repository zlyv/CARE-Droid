#!/usr/bin/env python3
from __future__ import annotations
import csv,math,json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score,confusion_matrix

ROOT=Path("artifacts/care-di25-compact43-relationquery-source-v1"); H0=Path("artifacts/care-di25-graphglobal-path-source-v1/arms/Q_QUERY_ONLY"); ARMS=("H0_LOCKED","C43_COMPACT","C43_RELATION_QUERY")
PATHS={"H0_LOCKED":H0,"C43_COMPACT":ROOT/"arms/C43_COMPACT","C43_RELATION_QUERY":ROOT/"arms/C43_RELATION_QUERY"}
METHOD={"H0_LOCKED":"DI25__Host__Anomaly__GraphGlobalPath__Q_QUERY_ONLY__Bounded","C43_COMPACT":"DI25__Compact43__Anomaly__QQuery__Bounded","C43_RELATION_QUERY":"DI25__Compact43__Anomaly__GraphGlobalRelationQuery__Bounded"}


def read(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def exact(a,b):
    n=a+b
    return 1.0 if not n else min(1.0,2*sum(math.comb(n,k) for k in range(min(a,b)+1))/(2**n))
def metric(y,p,base):
    pred=p>=.5;b=base>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();fixed=int(np.sum((b!=y)&(pred==y)));broken=int(np.sum((b==y)&(pred!=y)))
    return {"BA":float(balanced_accuracy_score(y,pred)),"AUC":float(roc_auc_score(y,p)),"F1":float(f1_score(y,pred)),"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),"fixed":fixed,"broken":broken,"net":fixed-broken}


def main():
    ref=read(ROOT/"preflight/di25_reference/source_predictions.csv");ref.sort(key=lambda r:r["sha256"].lower());ids=[r["sha256"].lower() for r in ref];y=np.asarray([int(float(r["label"])) for r in ref]);fold=np.asarray([int(float(r["fold"])) for r in ref]);route=np.asarray([int(float(r["A25__routed"])) for r in ref]);base=np.asarray([float(r["Global24__prob"]) for r in ref]);prob={};integrity=[]
    for arm in ARMS:
        rows=read(PATHS[arm]/"source_predictions.csv");by={r["sha256"].lower():r for r in rows};column=METHOD[arm]+"__prob"
        if set(by)!=set(ids):raise RuntimeError("BLOCKED_BY_CONTRACT: cohort "+arm)
        prob[arm]=np.asarray([float(by[s][column]) for s in ids])
        for i,s in enumerate(ids):
            actual=int(float(by[s]["A25__routed"]));integrity.append({"sha256":s,"arm":arm,"route_DI":int(route[i]),"route_arm":actual,"different":int(actual!=int(route[i]))})
    if sum(r["different"] for r in integrity):raise RuntimeError("BLOCKED_BY_CONTRACT: route changed")
    counts={}
    for arm in ARMS[1:]:
        fit=read(PATHS[arm]/"fit_audit.csv");values={int(float(r["parameter_n"])) for r in fit}
        if len(values)!=1:raise RuntimeError("BLOCKED_BY_CONTRACT: parameter inconsistency")
        counts[arm]=next(iter(values))
    if len(set(counts.values()))!=1:raise RuntimeError("BLOCKED_BY_CONTRACT: candidate capacity differs")
    aggregate=[{"arm":a,"parameter_count":33036 if a=="H0_LOCKED" else counts[a],**metric(y,prob[a],base)} for a in ARMS];foldrows=[{"arm":a,"fold":f,**metric(y[fold==f],prob[a][fold==f],base[fold==f])} for a in ARMS for f in range(1,6)];h0=(prob["H0_LOCKED"]>=.5)==y;paired=[]
    for arm in ARMS[1:]:
        c=(prob[arm]>=.5)==y;n01=int(np.sum(~h0&c));n10=int(np.sum(h0&~c));paired.append({"comparison":arm+"_minus_H0","candidate_only_correct":n01,"H0_only_correct":n10,"paired_net":n01-n10,"exact_p":exact(n01,n10)})
    write(ROOT/"results/aggregate_metrics.csv",aggregate);write(ROOT/"results/per_fold_metrics.csv",foldrows);write(ROOT/"results/paired_vs_H0.csv",paired);write(ROOT/"results/route_integrity.csv",integrity)
    lines=["CARE-Droid duplicate-free Compact43 relation-query source v1","","Strict 2017 OOF; frozen DI25=200/799; no future row or Ego tensor.","Compact43 stores SelfRole31+Tail12 once. Weighted background distance preserves H0's double role weighting.","Relation31 is instantiated in both candidate arms and can affect only Host attention in C43_RELATION_QUERY.",""]
    for r in aggregate:lines.append(f"{r['arm']}: BA={r['BA']:.6f} AUC={r['AUC']:.6f} F1={r['F1']:.6f} fixed={r['fixed']} broken={r['broken']} net={r['net']} parameters={r['parameter_count']}")
    lines+=["","Paired versus H0:"]+[f"{r['comparison']}: candidate_only={r['candidate_only_correct']} H0_only={r['H0_only_correct']} net={r['paired_net']} p={r['exact_p']:.6f}" for r in paired]
    lines+=["","Do not run future diagnostics until the source result is inspected."]
    (ROOT/"results/REPORT.txt").write_text("\n".join(lines)+"\n");print("COMPACT43_RELATIONQUERY_SOURCE_SUMMARY_PASS");print((ROOT/"results/REPORT.txt").read_text())


if __name__=="__main__":main()
