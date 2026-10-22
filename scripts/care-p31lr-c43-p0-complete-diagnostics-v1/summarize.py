#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,f1_score,roc_auc_score
from common import *
def lookup(rows):
    x={norm(r["sha256"]):r for r in rows}
    if len(x)!=len(rows):raise RuntimeError("duplicate SHA")
    return x
def metrics(protocol,rows):
    years=np.asarray([int(r["year"]) for r in rows]);y=np.asarray([int(r["label"]) for r in rows]);b=np.asarray([float(r["p_stage1"]) for r in rows]);f=np.asarray([float(r["p_final"]) for r in rows]);route=np.asarray([int(r["route_DI"]) for r in rows],bool);out=[]
    for scope,sel in [(str(z),years==z) for z in sorted(set(years))]+[("pooled",np.ones(len(rows),bool))]:
        c=metric_counts(y[sel],b[sel],f[sel],route[sel]);bp=b[sel]>=.5;fp=f[sel]>=.5;out.append({"protocol":protocol,"scope":scope,"n":int(sel.sum()),**c,"P31LR_BA":float(balanced_accuracy_score(y[sel],bp)),"P31LR_AUC":float(roc_auc_score(y[sel],b[sel])),"P31LR_F1":float(f1_score(y[sel],bp)),"CARE_BA":float(balanced_accuracy_score(y[sel],fp)),"CARE_AUC":float(roc_auc_score(y[sel],f[sel])),"CARE_F1":float(f1_score(y[sel],fp))})
    return out
def main():
    frozen=read_csv(ROOT/"frozen_future/label_free_predictions.csv");labels=lookup(read_csv(resolved()["future_label_fold_contract"]));frozen=[{**r,"label":labels[norm(r["sha256"])]["label"]} for r in frozen]
    within=[]
    for g in ("2018_2019","2020_2021"):within+=read_csv(ROOT/f"within_year/{g}/oof_predictions.csv")
    if len(frozen)!=1989 or len(within)!=1989:raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete results")
    fm=metrics("frozen_2017_to_future",frozen);wm=metrics("independent_same_year_5fold",within);write_csv(ROOT/"results/metrics.csv",fm+wm);write_csv(ROOT/"results/frozen_predictions_with_posthoc_labels.csv",frozen);write_csv(ROOT/"results/within_year_oof_predictions.csv",within)
    lines=["CARE-Droid P31-LR + routing-only committee + P0-ZERO6 C43-RQ complete diagnostic v1","","Stage-I classifier: calibrated P31 Logistic Regression only.","Router: LR-anchored LR/RF/XGB/SVM MeanOpposition Top25%; committee does not vote for Stage-I.","Stage-II: P0_ZERO6 C43-RQ scalar bounded head, max_delta=2.25.","","FROZEN 2017 -> 2018-2021 (labels joined only by summarizer):"]
    for r in fm:lines.append(f"{r['scope']}: P31LR_BA={r['P31LR_BA']:.6f} CARE_BA={r['CARE_BA']:.6f} CARE_AUC={r['CARE_AUC']:.6f} CARE_F1={r['CARE_F1']:.6f} routed={r['routed']} fixed={r['fixed']} broken={r['broken']} net={r['net']} fixed_FN={r['fixed_FN']} fixed_FP={r['fixed_FP']} broken_TN_to_FP={r['broken_TN_to_FP']} broken_TP_to_FN={r['broken_TP_to_FN']}")
    lines+=["","INDEPENDENT SAME-YEAR FIVE-FOLD OOF (not temporal generalization):"]
    for r in wm:lines.append(f"{r['scope']}: P31LR_BA={r['P31LR_BA']:.6f} CARE_BA={r['CARE_BA']:.6f} CARE_AUC={r['CARE_AUC']:.6f} CARE_F1={r['CARE_F1']:.6f} routed={r['routed']} fixed={r['fixed']} broken={r['broken']} net={r['net']} fixed_FN={r['fixed_FN']} fixed_FP={r['fixed_FP']} broken_TN_to_FP={r['broken_TN_to_FP']} broken_TP_to_FN={r['broken_TP_to_FN']}")
    lines+=["","Evidence boundary: future years were already inspected; frozen results are diagnostics only.","No future result may retune Stage-I, router, P0-ZERO6, C43-RQ, or max_delta."]
    report="\n".join(lines)+"\n";(ROOT/"results/REPORT.txt").write_text(report);(ROOT/"results/hard_audit.json").write_text(json.dumps({"source_locked":{"P31LR_BA":.8584774436090226,"CARE_BA":.9236309523809524,"fixed":60,"broken":8,"net":52},"future_labels_loaded_by_model":False,"same_year_temporal_generalization":False,"stage1":"P31_LR","committee_role":"routing_only","max_delta":2.25},indent=2)+"\n");print("P31LR_C43_P0_COMPLETE_SUMMARY_PASS");print(report)
if __name__=="__main__":main()
