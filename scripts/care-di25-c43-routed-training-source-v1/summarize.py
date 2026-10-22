#!/usr/bin/env python3
from __future__ import annotations
import csv,math
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score,confusion_matrix

ROOT=Path("artifacts/care-di25-c43-routed-training-source-v1");BASE=Path("artifacts/care-di25-compact43-relationquery-source-v1")
ARMS=("T0_ALL_TRAIN_LOCKED","T1_ROUTED_LOSS","T2_FULLY_ROUTED")
PATHS={"T0_ALL_TRAIN_LOCKED":BASE/"arms/C43_RELATION_QUERY","T1_ROUTED_LOSS":ROOT/"arms/T1_ROUTED_LOSS","T2_FULLY_ROUTED":ROOT/"arms/T2_FULLY_ROUTED"}
METHOD={"T0_ALL_TRAIN_LOCKED":"DI25__Compact43__Anomaly__GraphGlobalRelationQuery__Bounded","T1_ROUTED_LOSS":"DI25__C43RQ__RoutedLoss__Bounded","T2_FULLY_ROUTED":"DI25__C43RQ__FullyRoutedTrain__Bounded"}
def read(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def exact(a,b):
    n=a+b
    return 1.0 if not n else min(1.0,2*sum(math.comb(n,k) for k in range(min(a,b)+1))/(2**n))
def metric(y,p,base,route):
    pred=p>=.5;b=base>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();fixed=int(np.sum((b!=y)&(pred==y)));broken=int(np.sum((b==y)&(pred!=y)));rerr=route&(b!=y);rsafe=route&(b==y)
    return {"BA":float(balanced_accuracy_score(y,pred)),"AUC":float(roc_auc_score(y,p)),"F1":float(f1_score(y,pred,zero_division=0)),"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),"fixed":fixed,"broken":broken,"net":fixed-broken,"routed_error_n":int(rerr.sum()),"routed_safe_n":int(rsafe.sum()),"correction_recall":float(fixed/max(int(rerr.sum()),1)),"break_rate":float(broken/max(int(rsafe.sum()),1))}
def main():
    ref=read(ROOT/"preflight/di25_reference/source_predictions.csv");ref.sort(key=lambda r:r["sha256"].lower());ids=[r["sha256"].lower() for r in ref];y=np.asarray([int(float(r["label"])) for r in ref]);fold=np.asarray([int(float(r["fold"])) for r in ref]);route=np.asarray([int(float(r["A25__routed"]))==1 for r in ref]);base=np.asarray([float(r["Global24__prob"]) for r in ref]);prob={}
    for arm in ARMS:
        rows=read(PATHS[arm]/"source_predictions.csv");by={r["sha256"].lower():r for r in rows};column=METHOD[arm]+"__prob"
        if set(by)!=set(ids):raise RuntimeError("BLOCKED_BY_CONTRACT: cohort mismatch "+arm)
        prob[arm]=np.asarray([float(by[s][column]) for s in ids])
        if any(int(float(by[s]["A25__routed"]))!=int(route[i]) for i,s in enumerate(ids)):raise RuntimeError("BLOCKED_BY_CONTRACT: route changed "+arm)
    aggregate=[{"arm":arm,"parameter_count":32292,**metric(y,prob[arm],base,route)} for arm in ARMS]
    per_fold=[{"arm":arm,"fold":f,**metric(y[fold==f],prob[arm][fold==f],base[fold==f],route[fold==f])} for arm in ARMS for f in range(1,6)]
    control=(prob[ARMS[0]]>=.5)==y;paired=[]
    for arm in ARMS[1:]:
        correct=(prob[arm]>=.5)==y;n01=int(np.sum(~control&correct));n10=int(np.sum(control&~correct));paired.append({"comparison":arm+"_minus_T0","candidate_only_correct":n01,"T0_only_correct":n10,"paired_net":n01-n10,"exact_p":exact(n01,n10)})
    seed_metrics=[]
    for arm in ARMS[1:]:
        rows=read(PATHS[arm]/"per_seed_predictions.csv");by={r["sha256"].lower():r for r in rows}
        for seed in (0,1,2):seed_metrics.append({"arm":arm,"seed":seed,**metric(y,np.asarray([float(by[s][f"p_final_seed_{seed}"]) for s in ids]),base,route)})
    seed_summary=[]
    for arm in ARMS[1:]:
        part=[r for r in seed_metrics if r["arm"]==arm]
        seed_summary.append({"arm":arm,"BA_mean":float(np.mean([r["BA"] for r in part])),"BA_std":float(np.std([r["BA"] for r in part],ddof=1)),"fixed_mean":float(np.mean([r["fixed"] for r in part])),"fixed_min":min(r["fixed"] for r in part),"fixed_max":max(r["fixed"] for r in part),"broken_mean":float(np.mean([r["broken"] for r in part])),"broken_min":min(r["broken"] for r in part),"broken_max":max(r["broken"] for r in part),"net_mean":float(np.mean([r["net"] for r in part])),"net_min":min(r["net"] for r in part),"net_max":max(r["net"] for r in part)})
    write(ROOT/"results/aggregate_metrics.csv",aggregate);write(ROOT/"results/per_fold_metrics.csv",per_fold);write(ROOT/"results/paired_vs_T0.csv",paired);write(ROOT/"results/per_seed_metrics.csv",seed_metrics);write(ROOT/"results/seed_robustness.csv",seed_summary)
    lines=["CARE-Droid C43-RQ routed-training source ablation v1","","Strict 2017 five-fold OOF; frozen DI25=200/799; no future row or Ego.","T0 is locked and not retrained. T1/T2 reuse T0 selected epochs per fold and seed.","T1 changes only neural-loss scope. T2 additionally fits Host backgrounds and Stage-II graph scaler on routed outer-train APKs.",""]
    for r in aggregate:lines.append(f"{r['arm']}: BA={r['BA']:.6f} AUC={r['AUC']:.6f} F1={r['F1']:.6f} fixed={r['fixed']} broken={r['broken']} net={r['net']} correction_recall={r['correction_recall']:.6f} break_rate={r['break_rate']:.6f}")
    lines += ["","Paired versus locked T0:"]+[f"{r['comparison']}: candidate_only={r['candidate_only_correct']} T0_only={r['T0_only_correct']} net={r['paired_net']} p={r['exact_p']:.6f}" for r in paired]
    lines += ["","Seed robustness:"]+[f"{r['arm']}: BA={r['BA_mean']:.6f}±{r['BA_std']:.6f} fixed={r['fixed_mean']:.2f}[{r['fixed_min']},{r['fixed_max']}] broken={r['broken_mean']:.2f}[{r['broken_min']},{r['broken_max']}] net={r['net_mean']:.2f}[{r['net_min']},{r['net_max']}]" for r in seed_summary]
    lines += ["","Do not run future diagnostics until source utility, Broken, fold consistency, and seed variance are inspected."]
    (ROOT/"results/REPORT.txt").write_text("\n".join(lines)+"\n");print("C43_ROUTED_TRAINING_SOURCE_SUMMARY_PASS");print("\n".join(lines))
if __name__=="__main__":main()

