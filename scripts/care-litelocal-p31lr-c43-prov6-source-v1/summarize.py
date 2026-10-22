#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score,confusion_matrix
ROOT=Path("artifacts/care-litelocal-p31lr-c43-prov6-source-v1")
ARMS={"P0_ZERO6":"P31LR__Opp25__C43Prov6Zero__Bounded","P1_PROV6":"P31LR__Opp25__C43Prov6Real__Bounded","P2_SHUFFLED_PROV6":"P31LR__Opp25__C43Prov6Shuffled__Bounded"}
def read(p):
    with p.open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def write(p,rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def metrics(y,p):
    q=p>=.5;tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel();return {"BA":float(balanced_accuracy_score(y,q)),"AUC":float(roc_auc_score(y,p)),"F1":float(f1_score(y,q)),"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),"errors":int(np.sum(q!=y))}
def main():
    ref=sorted(read(ROOT/"preflight/lr_reference/source_predictions.csv"),key=lambda r:r["sha256"]);sha=[r["sha256"] for r in ref];y=np.asarray([int(r["label"]) for r in ref]);fold=np.asarray([int(r["fold"]) for r in ref]);base=np.asarray([float(r["P31_LR__prob"]) for r in ref]);route=np.asarray([int(r["A25__routed"])==1 for r in ref]);baseok=(base>=.5)==y
    aggregate=[{"arm":"P31_LR_BASE",**metrics(y,base),"fixed":0,"broken":0,"net":0,"fixed_FN":0,"fixed_FP":0,"broken_TN":0,"broken_TP":0,"routed":0}];fold_rows=[];probs={};oks={}
    for arm,method in ARMS.items():
        rows=sorted(read(ROOT/"arms"/arm/"source_predictions.csv"),key=lambda r:r["sha256"])
        if [r["sha256"] for r in rows]!=sha:raise RuntimeError("BLOCKED_BY_CONTRACT: arm cohort mismatch")
        observed=np.asarray([float(r["Global24__prob"]) for r in rows]);routed=np.asarray([int(r["A25__routed"])==1 for r in rows]);p=np.asarray([float(r[method+"__prob"]) for r in rows])
        if np.max(np.abs(observed-base))>1e-12 or np.any(routed!=route) or np.max(np.abs(p[~route]-base[~route]))>1e-12:raise RuntimeError("BLOCKED_BY_CONTRACT: base/route/outside-route changed")
        ok=(p>=.5)==y;fixed=~baseok&ok;broken=baseok&~ok
        row={"arm":arm,**metrics(y,p),"fixed":int(fixed.sum()),"broken":int(broken.sum()),"net":int(fixed.sum()-broken.sum()),"fixed_FN":int(np.sum(fixed&(y==1))),"fixed_FP":int(np.sum(fixed&(y==0))),"broken_TN":int(np.sum(broken&(y==0))),"broken_TP":int(np.sum(broken&(y==1))),"routed":200};aggregate.append(row);probs[arm]=p;oks[arm]=ok
        fit=read(ROOT/"arms"/arm/"fit_audit.csv");counts={int(float(r["parameter_n"])) for r in fit}
        if len(fit)!=15 or len(counts)!=1:raise RuntimeError("BLOCKED_BY_CONTRACT: fit audit failed")
        row["parameter_count"]=next(iter(counts))
        for f in range(1,6):
            s=fold==f;fx=~baseok[s]&ok[s];br=baseok[s]&~ok[s];fold_rows.append({"arm":arm,"fold":f,**metrics(y[s],p[s]),"fixed":int(fx.sum()),"broken":int(br.sum()),"net":int(fx.sum()-br.sum())})
    parameter_counts={r["parameter_count"] for r in aggregate if r["arm"]!="P31_LR_BASE"}
    if len(parameter_counts)!=1:raise RuntimeError("BLOCKED_BY_CONTRACT: arms are not capacity-matched")
    paired=[]
    for control in ("P0_ZERO6","P2_SHUFFLED_PROV6"):
        only=np.sum(oks["P1_PROV6"]&~oks[control]);other=np.sum(~oks["P1_PROV6"]&oks[control]);pval=float(binomtest(min(only,other),only+other,.5).pvalue) if only+other else 1.
        paired.append({"comparison":"P1_PROV6_minus_"+control,"P1_only_correct":int(only),"control_only_correct":int(other),"paired_net":int(only-other),"exact_p":pval})
    a={r["arm"]:r for r in aggregate};positive=sum(next(r for r in fold_rows if r["arm"]=="P1_PROV6" and r["fold"]==f)["net"]>=next(r for r in fold_rows if r["arm"]=="P0_ZERO6" and r["fold"]==f)["net"] for f in range(1,6))
    advance=a["P1_PROV6"]["net"]>a["P0_ZERO6"]["net"] and a["P1_PROV6"]["broken"]<=a["P0_ZERO6"]["broken"] and a["P1_PROV6"]["fixed_FP"]>a["P0_ZERO6"]["fixed_FP"] and paired[1]["paired_net"]>0 and positive>=3
    out=ROOT/"results";write(out/"aggregate_metrics.csv",aggregate);write(out/"per_fold_metrics.csv",fold_rows);write(out/"paired_comparisons.csv",paired)
    audit={"source_rows":799,"year":2017,"stage1":"P31-LR only","committee_role":"routing only","route":200,"route_difference_count":0,"parameter_count":next(iter(parameter_counts)),"all_arms_capacity_matched":True,"P1_nonnegative_net_vs_P0_folds":int(positive),"advance_candidate":bool(advance),"future_rows_loaded":0,"ego_consumed":False};(out/"hard_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    lines=["CARE-Droid P31-LR C43-RQ Provenance6 source v1","","Strict 2017 five-fold OOF; P31-LR is the sole Stage-I classifier; LR/RF/XGB/SVM are routing-only.","Frozen foldwise 25% LR-anchored opposition route=200/799; C43-RQ scalar bounded head unchanged.","P0/P1/P2 are capacity-matched; P2 shuffles Prov6 within APK and primary role. No Ego/future row.",""]
    for r in aggregate:lines.append(f"{r['arm']}: BA={r['BA']:.6f} AUC={r['AUC']:.6f} F1={r['F1']:.6f} fixed={r['fixed']} broken={r['broken']} net={r['net']} fixed_FN={r['fixed_FN']} fixed_FP={r['fixed_FP']}")
    lines += ["",*(f"{r['comparison']}: P1_only={r['P1_only_correct']} control_only={r['control_only_correct']} net={r['paired_net']} p={r['exact_p']:.6f}" for r in paired),"",f"P1_nonnegative_net_vs_P0_folds={positive}/5",f"SOURCE_ADVANCE_CANDIDATE={'P1_PROV6' if advance else 'NONE'}","Human approval is required before any future diagnostic. This source experiment does not authorize Action3."]
    (out/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8");print("P31LR_PROV6_SOURCE_SUMMARY_PASS");print("\n".join(lines))
if __name__=="__main__":main()
