#!/usr/bin/env python3
from __future__ import annotations
import json
import numpy as np
from common import ROOT,RANDOM_SEEDS,BUDGETS,read_csv,write_csv,dump,metrics,deterministic_route

rows=read_csv(ROOT/"formal_predictions.csv");n=len(rows);sha=np.asarray([r["sha256"] for r in rows]);fold=np.asarray([int(r["fold"]) for r in rows]);y=np.asarray([int(r["label"]) for r in rows]);base=np.asarray([float(r["stage1_prob"]) for r in rows]);formal_route=np.asarray([int(r["routed"]) for r in rows],bool);router_score=np.asarray([float(r["router_score"]) for r in rows]);candidate=np.asarray([float(r["bounded_prob"]) for r in rows]);formal=np.asarray([float(r["final_prob"]) for r in rows])

def arm_rows(arm,p,route,seed="",aggregate_type="pooled"):
    row={"arm":arm,"seed":seed,"aggregate_type":aggregate_type,"deep_analysis_count":int(np.sum(route)),"deep_analysis_ratio":float(np.mean(route)),**metrics(y,base,p,route)}
    folds=[]
    for f in range(1,6):
        z=fold==f;folds.append({"arm":arm,"seed":seed,"aggregate_type":"fold","fold":f,"deep_analysis_count":int(np.sum(route[z])),"deep_analysis_ratio":float(np.mean(route[z])),**metrics(y[z],base[z],p[z],route[z])})
    return row,folds

aggregate=[];fold_results=[]
a0,fr=arm_rows("A0_STAGE1_ONLY",base,np.zeros(n,bool));aggregate.append(a0);fold_results+=fr
a2,fr=arm_rows("A2_DI25_FORMAL",formal,formal_route);aggregate.append(a2);fold_results+=fr
a3route=np.ones(n,bool);a3,fr=arm_rows("A3_FULL100",candidate,a3route);aggregate.append(a3);fold_results+=fr

random_rows=[];random_routes={}
for seed in RANDOM_SEEDS:
    rr=np.zeros(n,bool)
    for f in range(1,6):
        idx=np.flatnonzero(fold==f);count=int(np.sum(formal_route[idx]));rr|=deterministic_route(idx,sha,count,seed+f*1009)
    if [int(np.sum(rr&(fold==f))) for f in range(1,6)]!=[40]*5:raise RuntimeError("random route count mismatch")
    p=np.where(rr,candidate,base);row,fr=arm_rows("A1_RANDOM25",p,rr,seed);random_rows.append(row);fold_results+=fr;random_routes[seed]=rr
write_csv(ROOT/"random25_seed_results.csv",random_rows)
numeric=("BA","AUC","F1","TN","FP","FN","TP","wrong","fixed","broken","net","routed","prediction_class_flips","deep_analysis_count","deep_analysis_ratio")
mean={"arm":"A1_RANDOM25","seed":"MEAN","aggregate_type":"mean_across_5_frozen_seeds"};std={"arm":"A1_RANDOM25_STD","seed":"STD","aggregate_type":"std_across_5_frozen_seeds"}
for key in numeric:
    values=[float(r[key]) for r in random_rows];mean[key]=float(np.mean(values));std[key]=float(np.std(values))
aggregate.insert(1,mean);aggregate.insert(2,std)
for f in range(1,6):
    z=[r for r in fold_results if r["arm"]=="A1_RANDOM25" and int(r["fold"])==f and r["seed"] not in {"MEAN","STD"}]
    fm={"arm":"A1_RANDOM25","seed":"MEAN","aggregate_type":"fold_mean_across_5_frozen_seeds","fold":f};fs={"arm":"A1_RANDOM25","seed":"STD","aggregate_type":"fold_std_across_5_frozen_seeds","fold":f}
    for key in numeric:
        values=[float(r[key]) for r in z];fm[key]=float(np.mean(values));fs[key]=float(np.std(values))
    fold_results.extend((fm,fs))
write_csv(ROOT/"formal_ablation_results.csv",aggregate);write_csv(ROOT/"formal_ablation_fold_results.csv",fold_results)

def routing_row(router,seed,f,route_mask):
    z=fold==f;r=route_mask&z;errors=z&((base>=.5)!=y);re=int(np.sum(r&errors));total_error=int(np.sum(errors));routed_n=int(np.sum(r));unrouted_n=int(np.sum(z&~route_mask));unrouted_error=total_error-re;precision=re/max(routed_n,1);overall=total_error/max(int(np.sum(z)),1)
    return {"router":router,"seed":seed,"fold":f,"total_samples":int(np.sum(z)),"routed_samples":routed_n,"route_fraction":routed_n/int(np.sum(z)),"stage1_errors":total_error,"routed_stage1_errors":re,"unrouted_stage1_errors":unrouted_error,"error_recall":re/max(total_error,1),"routing_precision":precision,"error_enrichment":precision/max(overall,1e-12),"accepted_risk":unrouted_error/max(unrouted_n,1)}
routing=[]
for f in range(1,6):routing.append(routing_row("DI25","FORMAL",f,formal_route))
for seed,rr in random_routes.items():
    for f in range(1,6):routing.append(routing_row("RANDOM25",seed,f,rr))
write_csv(ROOT/"routing_analysis.csv",routing)

budget_rows=[]
for budget in BUDGETS:
    rr=np.zeros(n,bool)
    for f in range(1,6):
        idx=np.flatnonzero(fold==f);count=int(round(len(idx)*budget));order=np.lexsort((sha[idx],-router_score[idx]));rr[idx[order[:count]]]=True
    if abs(budget-.25)<1e-12 and not np.array_equal(rr,formal_route):raise RuntimeError("BLOCKED_BY_CONTRACT: continuous router score does not reconstruct formal DI25 route")
    p=np.where(rr,candidate,base);m=metrics(y,base,p,rr);errors=(base>=.5)!=y;unrouted=~rr
    budget_rows.append({"budget_percent":int(budget*100),"formal_budget":int(abs(budget-.25)<1e-12),"used_for_selection":0,"routed_count":int(rr.sum()),"route_fraction":float(rr.mean()),"stage1_errors":int(errors.sum()),"routed_stage1_errors":int(np.sum(rr&errors)),"error_recall":float(np.sum(rr&errors)/max(errors.sum(),1)),"accepted_risk":float(np.sum(unrouted&errors)/max(unrouted.sum(),1)),**m})
write_csv(ROOT/"budget_curve.csv",budget_rows)

change=[]
for i in range(n):
    if abs(formal[i]-base[i])<=1e-15:continue
    bp=int(base[i]>=.5);fp=int(formal[i]>=.5);base_ok=bp==y[i];final_ok=fp==y[i]
    outcome="FIXED" if (not base_ok and final_ok) else ("BROKEN" if (base_ok and not final_ok) else ("PROBABILITY_CHANGED_CORRECT" if final_ok else "PROBABILITY_CHANGED_WRONG"))
    change.append({"sha256":sha[i],"fold":fold[i],"true_label":y[i],"stage1_prob":base[i],"stage1_pred":bp,"router_score":router_score[i],"routed":int(formal_route[i]),"candidate_prob":candidate[i],"bounded_prob":candidate[i],"final_prob":formal[i],"final_pred":fp,"classification_flipped":int(bp!=fp),"outcome":outcome})
write_csv(ROOT/"formal_prediction_change_audit.csv",change)
if sum(r["outcome"]=="FIXED" for r in change)!=11 or sum(r["outcome"]=="BROKEN" for r in change)!=7:raise RuntimeError("formal fixed/broken reconstruction failed")
dump(ROOT/"bounded_ablation_audit.json",{"status":"A4_NOT_RUN_NO_CANONICAL_PREBOUND_DECISION","reason":"only seed-averaged bounded_delta and bounded candidate probability are saved; no unique pre-bound prediction exists","future_rows_loaded":0})
print("A0_STAGE1_ONLY_DONE");print("A1_RANDOM25_DONE");print("A2_DI25_FORMAL_DONE");print("A3_FULL100_DONE");print("A4_NOT_RUN_NO_CANONICAL_PREBOUND_DECISION");print("ROUTING_ANALYSIS_DONE");print("BUDGET_CURVE_DONE")

