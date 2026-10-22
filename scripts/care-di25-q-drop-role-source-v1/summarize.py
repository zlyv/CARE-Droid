#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score,confusion_matrix
ROOT=Path("artifacts/care-di25-q-drop-role-source-v1");FULL=Path("artifacts/care-di25-graphglobal-path-source-v1/arms/Q_QUERY_ONLY/source_predictions.csv");CAND=ROOT/"arm/source_predictions.csv"
def read(p):
    with p.open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def col(r):
    c=[x for x in r[0] if x.endswith("__Bounded__prob")]
    if len(c)!=1:raise RuntimeError("probability column mismatch "+repr(c))
    return c[0]
def met(y,p,b):
    z=p>=.5;bz=b>=.5;tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel();fx=int(np.sum((z==y)&(bz!=y)));br=int(np.sum((z!=y)&(bz==y)))
    return dict(BA=float(balanced_accuracy_score(y,z)),AUC=float(roc_auc_score(y,p)),F1=float(f1_score(y,z)),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),fixed=fx,broken=br,net=fx-br)
def main():
    f=read(FULL);c=read(CAND);ids=[r["sha256"].lower() for r in f];cm={r["sha256"].lower():r for r in c}
    if len(f)!=799 or len(c)!=799 or set(ids)!=set(cm):raise RuntimeError("BLOCKED_BY_CONTRACT: cohort mismatch")
    c=[cm[i] for i in ids];y=np.array([int(float(r["label"])) for r in f]);b=np.array([float(r["Global24__prob"]) for r in f]);route=np.array([int(float(r["A25__routed"])) for r in f])
    if route.sum()!=200 or any(int(float(c[i]["A25__routed"]))!=route[i] for i in range(799)):raise RuntimeError("BLOCKED_BY_CONTRACT: route mismatch")
    pf=np.array([float(r[col(f)]) for r in f]);pc=np.array([float(r[col(c)]) for r in c]);mf=met(y,pf,b);mc=met(y,pc,b)
    cf=(pf>=.5)==y;cc=(pc>=.5)==y;n01=int(np.sum((~cf)&cc));n10=int(np.sum(cf&(~cc)));pv=float(binomtest(min(n01,n10),n01+n10,.5).pvalue) if n01+n10 else 1.0
    folds=np.array([int(float(r["fold"])) for r in f]);fr=[]
    for k in range(1,6):
        z=folds==k
        for name,p in (("FULL_Q",pf),("Q_DROP_ROLE_DIRECT",pc)):fr.append({"arm":name,"fold":k,**met(y[z],p[z],b[z])})
    ROOT.joinpath("results").mkdir(parents=True,exist_ok=True)
    with (ROOT/"results/fold_metrics.csv").open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=list(fr[0]));w.writeheader();w.writerows(fr)
    result={"FULL_Q":mf,"Q_DROP_ROLE_DIRECT":mc,"candidate_only_correct":n01,"FULL_Q_only_correct":n10,"paired_net":n01-n10,"exact_p":pv,"route_difference_count":0,"parameter_count":33036,"future_rows_loaded":0}
    lines=["CARE-Droid Q-Query drop-direct-RoleToken source experiment v1","","Strict 2017 OOF; frozen DI25=200/799; Host-only; no Ego/future row.","RoleToken and GraphGlobal still localize Host; only direct RoleToken -> delta is masked.","FULL_Q is locked and not retrained.","",f"FULL_Q: BA={mf['BA']:.6f} AUC={mf['AUC']:.6f} F1={mf['F1']:.6f} fixed={mf['fixed']} broken={mf['broken']} net={mf['net']}",f"Q_DROP_ROLE_DIRECT: BA={mc['BA']:.6f} AUC={mc['AUC']:.6f} F1={mc['F1']:.6f} fixed={mc['fixed']} broken={mc['broken']} net={mc['net']}","",f"Paired candidate-only={n01} FULL_Q-only={n10} net={n01-n10} exact_p={pv:.6f}","Do not run future evaluation until the source result is inspected and explicitly frozen."]
    (ROOT/"results/REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8");(ROOT/"results/summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("Q_DROP_ROLE_SOURCE_SUMMARY_PASS");print((ROOT/"results/REPORT.txt").read_text())
if __name__=="__main__":main()
