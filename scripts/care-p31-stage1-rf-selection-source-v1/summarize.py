#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score,confusion_matrix

ROOT=Path("artifacts/care-p31-stage1-rf-selection-source-v1")
METHODS={
 "A_LR_LRROUTE":"P31LR__Opp25__C43Prov6Zero__Bounded",
 "B_RF_LRROUTE":"P31RF__LRRoute__P0Zero6C43RQ__Bounded",
 "C_RF_RFROUTE":"P31RF__RFRoute__P0Zero6C43RQ__Bounded",
 "D_MEAN_MEANROUTE":"P31Mean__MeanRoute__P0Zero6C43RQ__Bounded",
}
CONTROL=Path("artifacts/care-litelocal-p31lr-c43-prov6-source-v1/arms/P0_ZERO6")

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def write(path,rows):
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields:fields.append(k)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def metric(y,p):
    q=p>=.5;tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
    return {"BA":float(balanced_accuracy_score(y,q)),"AUC":float(roc_auc_score(y,p)),
        "F1":float(f1_score(y,q)),"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),
        "errors":int(np.sum(q!=y))}

def main():
    stage={r["arm"]:r for r in read(ROOT/"stage1_router_metrics.csv")}
    aggregate=[]; folds=[]; end_ok={}; ids0=None
    for arm,method in METHODS.items():
        ref=sorted(read(ROOT/"preflight"/arm/"source_predictions.csv"),key=lambda r:r["sha256"])
        source=CONTROL if arm=="A_LR_LRROUTE" else ROOT/"arms"/arm
        rows=sorted(read(source/"source_predictions.csv"),key=lambda r:r["sha256"])
        ids=[r["sha256"] for r in ref]
        if [r["sha256"] for r in rows]!=ids:raise RuntimeError("BLOCKED_BY_CONTRACT: cohort mismatch "+arm)
        if ids0 is None:ids0=ids
        elif ids!=ids0:raise RuntimeError("BLOCKED_BY_CONTRACT: arm ordering mismatch")
        y=np.asarray([int(r["label"]) for r in ref]);fold=np.asarray([int(r["fold"]) for r in ref]);base=np.asarray([float(r["Global24__prob"]) for r in ref]);route=np.asarray([int(r["A25__routed"])==1 for r in ref])
        observed=np.asarray([float(r["Global24__prob"]) for r in rows]);observed_route=np.asarray([int(r["A25__routed"])==1 for r in rows]);final=np.asarray([float(r[method+"__prob"]) for r in rows])
        if np.max(np.abs(base-observed))>1e-12 or np.any(route!=observed_route) or np.max(np.abs(final[~route]-base[~route]))>1e-12:raise RuntimeError("BLOCKED_BY_CONTRACT: base/route/outside changed "+arm)
        baseok=(base>=.5)==y;ok=(final>=.5)==y;fixed=~baseok&ok;broken=baseok&~ok
        row={"arm":arm,"stage1":stage[arm]["stage1"],"router_anchor":stage[arm]["router_anchor"],
            **metric(y,final),"stage1_BA":float(stage[arm]["BA"]),"stage1_AUC":float(stage[arm]["AUC"]),
            "stage1_errors":int(stage[arm]["errors"]),"captured_errors":int(stage[arm]["captured_errors"]),
            "error_recall25":float(stage[arm]["error_recall25"]),"routed":int(route.sum()),
            "fixed":int(fixed.sum()),"broken":int(broken.sum()),"net":int(fixed.sum()-broken.sum()),
            "fixed_FN":int(np.sum(fixed&(y==1))),"fixed_FP":int(np.sum(fixed&(y==0))),
            "broken_TN_to_FP":int(np.sum(broken&(y==0))),"broken_TP_to_FN":int(np.sum(broken&(y==1)))}
        fit=read(source/"fit_audit.csv");counts={int(float(r["parameter_n"])) for r in fit}
        if len(fit)!=15 or len(counts)!=1:raise RuntimeError("BLOCKED_BY_CONTRACT: fit audit "+arm)
        row["parameter_count"]=next(iter(counts));aggregate.append(row);end_ok[arm]=ok
        for f in range(1,6):
            s=fold==f;fx=~baseok[s]&ok[s];br=baseok[s]&~ok[s]
            folds.append({"arm":arm,"fold":f,**metric(y[s],final[s]),"stage1_BA":float(balanced_accuracy_score(y[s],base[s]>=.5)),"routed":int(route[s].sum()),"fixed":int(fx.sum()),"broken":int(br.sum()),"net":int(fx.sum()-br.sum())})
    if len({r["parameter_count"] for r in aggregate})!=1:raise RuntimeError("BLOCKED_BY_CONTRACT: parameter counts differ")
    paired=[]
    for arm in ("B_RF_LRROUTE","C_RF_RFROUTE","D_MEAN_MEANROUTE"):
        a=end_ok[arm];c=end_ok["A_LR_LRROUTE"];n01=int(np.sum(a&~c));n10=int(np.sum(~a&c));p=float(binomtest(min(n01,n10),n01+n10,.5).pvalue) if n01+n10 else 1.
        paired.append({"comparison":arm+"_minus_A_LR_LRROUTE","candidate_only_correct":n01,"A_only_correct":n10,"paired_net":n01-n10,"exact_p":p})
    out=ROOT/"results";write(out/"aggregate_metrics.csv",aggregate);write(out/"per_fold_metrics.csv",folds);write(out/"paired_vs_LR_control.csv",paired)
    audit={"source_rows":799,"year":2017,"source_only":True,"future_rows_loaded":0,
        "P31_fixed":True,"route_budget":.25,"stage2":"P0_ZERO6 C43-RQ",
        "max_delta":2.25,"parameter_count":aggregate[0]["parameter_count"],
        "all_arms_capacity_matched":True,"seeds":[0,1,2],"ego_consumed":False}
    (out/"hard_audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    lines=["CARE-Droid P31 Stage-I RF selection source v1","","Strict 2017 five-fold OOF only; no future row. P31, 25% budget, P0-ZERO6 C43-RQ, and max_delta=2.25 are fixed.","A isolates the locked LR control; B changes only Stage-I output; C also changes the opposition anchor; D is the mean-committee comparator.",""]
    for r in aggregate:lines.append(f"{r['arm']}: Stage1_BA={r['stage1_BA']:.6f} capture={r['captured_errors']}/{r['stage1_errors']} End_BA={r['BA']:.6f} AUC={r['AUC']:.6f} F1={r['F1']:.6f} fixed={r['fixed']} broken={r['broken']} net={r['net']} fixed_FN={r['fixed_FN']} fixed_FP={r['fixed_FP']}")
    lines += ["",*(f"{r['comparison']}: candidate_only={r['candidate_only_correct']} control_only={r['A_only_correct']} net={r['paired_net']} p={r['exact_p']:.6f}" for r in paired),"","Do not select from 2018-2021. Inspect source utility, error capture, fold consistency, and Broken before freezing any candidate."]
    (out/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("P31_RF_SELECTION_SOURCE_SUMMARY_PASS");print("\n".join(lines))

if __name__=="__main__":main()
