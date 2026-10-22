#!/usr/bin/env python3
"""Strict 2017 outer OOF with outer-train-only gate-threshold selection."""
from __future__ import annotations

import csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent; PROJECT=HERE.parent.parent
SOURCE=PROJECT/"scripts/care-di25-noego-needgate-source-v1"
for p in (HERE,SOURCE,PROJECT/"scripts/care-rolegraph-conditioned-v4-script",PROJECT/"scripts/care-rgcondattn-dedup-bounded-source-v1",PROJECT/"scripts/care-stage2only-anomaly-v1/vendor"):
    sys.path.insert(0,str(p))
import run_needgate_source as base
from fold_protocol import inner_oof_stage1, outer_fold_indices
from noego_representation import fit_seed
from probe_protocol import need_correct, stage1_logit
from source_loader import load_source_contract
from safety_core import apply, choose, fixed_broken
import run_care_rolegraph_conditioned_dualstream_v4 as v4
import run_care_rgcondattn_dedup_bounded_source_v1 as anomaly

OUT=Path("artifacts/care-di25-noego-safe-authority-full-v1/source")
SEEDS=(0,1,2)

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def probe(x,y,xv,seed):
    sc=StandardScaler().fit(x);m=LogisticRegression(C=1,class_weight="balanced",solver="liblinear",max_iter=5000,random_state=seed)
    m.fit(sc.transform(x),y);return m.predict_proba(sc.transform(xv))[:,1]

def gate_oof(x,y,seed):
    y=np.asarray(y); out=np.full(len(y),np.nan); splitter=StratifiedKFold(4,shuffle=True,random_state=seed)
    for tr,va in splitter.split(x,y): out[va]=probe(x[tr],y[tr],x[va],seed+len(va))
    if not np.isfinite(out).all(): raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete inner gate OOF")
    return out

def metrics(y,p):
    pred=p>=.5
    return {"BA":float(balanced_accuracy_score(y,pred)),"AUC":float(roc_auc_score(y,p)),"F1":float(f1_score(y,pred,zero_division=0)),
            "TN":int(np.sum((y==0)&(~pred))),"FP":int(np.sum((y==0)&pred)),"FN":int(np.sum((y==1)&(~pred))),"TP":int(np.sum((y==1)&pred))}

def main():
    args=base.arguments();base.assert_frozen(args);data=load_source_contract()
    if data.future_rows_loaded or data.ego_consumed or len(data.ids)!=799 or int(data.route.sum())!=200: raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort")
    device=v4.resolve_device(args.device); compact,names=base.compact_context(data);s7=base.source_s7_in_memory(data,args)
    v4.AUGMENTED_REGION_DIM=v4.v3.v2.BASE_REGION_DIM+4;v4.RoleGraphConditionedEvidence=anomaly.DedupBoundedRGCondAttn
    q_all=np.full(799,np.nan);p_final=np.asarray(data.p_stage1).copy();active_all=np.zeros(799,bool);limited_all=np.zeros(799,bool)
    threshold_rows=[];representation_audit=[]
    for fold in range(1,6):
        train,val=outer_fold_indices(data.folds,fold);rt=data.route[train];rv=data.route[val]
        gx,gval=base.fold_global24(data,compact,names,train,val)
        def predict(fit_idx,val_idx,seed):
            a,b=base.fold_global24(data,compact,names,fit_idx,val_idx);return v4.g24.fit_view(a,b,data.labels[fit_idx],np.arange(24),int(seed))
        bp=inner_oof_stage1(train,data.labels,predict,folds=4,split_seed=args.split_seed+1000*fold,model_seed=args.split_seed+10000*fold)
        bv=data.p_stage1[val]; yt=need_correct(data.labels[train],bp); yv=need_correct(data.labels[val],bv)
        lt=stage1_logit(bp);lv=stage1_logit(bv); inner_scores=[]; val_scores=[]
        for seed0 in SEEDS:
            seed=100000*fold+seed0
            tr_rep,va_rep,audit=fit_seed(v4,anomaly,gx,s7[train],s7[val],[data.host_bags[i] for i in train],[data.host_bags[i] for i in val],bp,data.labels[train],args,device,seed)
            representation_audit.append({"fold":fold,**audit})
            xtr=np.concatenate([tr_rep,lt[:,None]],axis=1);xva=np.concatenate([va_rep,lv[:,None]],axis=1)
            inner_scores.append(gate_oof(xtr[rt],yt[rt],args.split_seed+seed))
            val_scores.append(probe(xtr[rt],yt[rt],xva[rv],args.split_seed+seed))
        qi=np.mean(inner_scores,axis=0);qv=np.mean(val_scores,axis=0)
        best,_=choose(data.labels[train][rt],bp[rt],qi,np.ones(int(rt.sum()),bool))
        local_score=np.zeros(len(val));local_score[rv]=qv
        final,active,limited,_=apply(bv,local_score,rv,best["threshold"])
        p_final[val]=final;active_all[val]=active;limited_all[val]=limited;q_all[val[rv]]=qv
        fixed,broken=fixed_broken(data.labels[val],bv,final)
        threshold_rows.append({"fold":fold,"selected_threshold":best["threshold"],"outer_train_routed":int(rt.sum()),"outer_validation_routed":int(rv.sum()),
            "selection_fixed":best["fixed"],"selection_broken":best["broken"],"selection_utility":best["utility"],
            "validation_fixed":fixed,"validation_broken":broken,"validation_net":fixed-broken,
            "validation_gate_active":int(active.sum()),"future_rows_loaded":0,"ego_consumed":False})
        print(f"nested source fold {fold}/5 threshold={best['threshold']:.2f} done",flush=True)
    if not np.isfinite(q_all[data.route]).all() or np.isfinite(q_all[~data.route]).any(): raise RuntimeError("BLOCKED_BY_CONTRACT: OOF score coverage")
    deploy,_=choose(data.labels[data.route],data.p_stage1[data.route],q_all[data.route],np.ones(200,bool))
    fixed,broken=fixed_broken(data.labels,data.p_stage1,p_final);m=metrics(data.labels,p_final)
    rows=[]
    for i,sha in enumerate(data.ids): rows.append({"sha256":sha,"year":2017,"fold":int(data.folds[i]),"label":int(data.labels[i]),"route_DI":int(data.route[i]),
        "p_stage1":float(data.p_stage1[i]),"score_S3_nested":float(q_all[i]) if data.route[i] else "","selected_threshold":next(r["selected_threshold"] for r in threshold_rows if r["fold"]==int(data.folds[i])),
        "gate_active":int(active_all[i]),"p_final":float(p_final[i]),"bound_limited":int(limited_all[i])})
    write_csv(OUT/"nested_oof_predictions.csv",rows);write_csv(OUT/"threshold_by_fold.csv",threshold_rows);write_csv(OUT/"representation_audit.csv",representation_audit)
    config={"deploy_threshold":deploy["threshold"],"selection_utility":"Fixed-1.5*Broken","source_rows":799,"source_route_DI":200,"future_rows_loaded":0,"ego_consumed":False,
            "nested_source_metrics":{**m,"fixed":fixed,"broken":broken,"net":fixed-broken},"thresholds_by_fold":[r["selected_threshold"] for r in threshold_rows]}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"frozen_deploy_config.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    report=f"""CARE-Droid strict nested source no-Ego safe authority v1
Global24: BA=0.819674 AUC=0.908371 F1=0.803815
Nested Safe-S3: BA={m['BA']:.6f} AUC={m['AUC']:.6f} F1={m['F1']:.6f} fixed={fixed} broken={broken} net={fixed-broken}
outer-fold thresholds={config['thresholds_by_fold']}
source-selected frozen deploy threshold={deploy['threshold']:.2f}
Future rows loaded: 0. Ego consumed: false.
"""
    (OUT/"REPORT.txt").write_text(report,encoding="utf-8");print(report);print("NESTED_SOURCE_SAFE_AUTHORITY_DONE")

if __name__=="__main__":main()

