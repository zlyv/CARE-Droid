#!/usr/bin/env python3
"""Strict source OOF evaluation and shortcut audit for LiteSemantic."""
from __future__ import annotations
import csv,json,math,time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,confusion_matrix,f1_score,roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from feature_contract import arm_indices

ROOT=Path("artifacts/care-litesemantic-noshortcut-source-v1")
MEMBERS=("LR","RF","XGB","SVM")

def read(p):
    with Path(p).open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def norm(v):return str(v or "").strip().lower()
def write(p,rows):
    if not rows:return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with p.open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def make(name,seed,jobs=4):
    if name=="LR":return Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,solver="liblinear",max_iter=5000,random_state=seed))])
    if name=="RF":return RandomForestClassifier(n_estimators=300,max_features="sqrt",min_samples_leaf=2,class_weight="balanced",n_jobs=jobs,random_state=seed)
    if name=="XGB":return XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,reg_lambda=1,objective="binary:logistic",eval_metric="logloss",tree_method="hist",n_jobs=jobs,random_state=seed,verbosity=0)
    if name=="SVM":return Pipeline([("s",StandardScaler()),("m",SVC(C=1,kernel="rbf",gamma="scale",probability=False,random_state=seed))])
    raise KeyError(name)
def raw(model,x):
    if hasattr(model,"decision_function"):return np.asarray(model.decision_function(x),float).reshape(-1)
    p=np.clip(model.predict_proba(x)[:,list(model.classes_).index(1)],1e-8,1-1e-8);return np.log(p/(1-p))
def committee(xtr,ytr,xva,seed,jobs=4):
    splits=list(StratifiedKFold(4,shuffle=True,random_state=seed).split(xtr,ytr));out=np.zeros((len(xva),4));aud=[]
    for j,name in enumerate(MEMBERS):
        inner=np.full(len(xtr),np.nan)
        for k,(fit,cal) in enumerate(splits,1):
            m=make(name,seed+10000*j+100*k,jobs);m.fit(xtr[fit],ytr[fit]);inner[cal]=raw(m,xtr[cal])
        platt=LogisticRegression(C=1e6,solver="lbfgs",max_iter=2000,random_state=seed+10000*j+999);platt.fit(inner.reshape(-1,1),ytr)
        m=make(name,seed+10000*j,jobs);m.fit(xtr,ytr);out[:,j]=np.clip(platt.predict_proba(raw(m,xva).reshape(-1,1))[:,1],1e-8,1-1e-8)
        aud.append({"member":name,"dimension":xtr.shape[1],"train_n":len(xtr),"validation_n":len(xva),"protocol_seed":seed,"target_labels_received":False})
    return out,aud
def top_foldwise(score,folds,ids):
    route=np.zeros(len(score),bool)
    for f in sorted(set(folds)):
        q=np.flatnonzero(folds==f);n=int(round(len(q)*.25));order=np.lexsort((ids[q],-score[q]));route[q[order[:n]]]=1
    return route
def classification(y,p):
    pred=p>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();return {"BA":float(balanced_accuracy_score(y,pred)),"AUC":float(roc_auc_score(y,p)),"F1":float(f1_score(y,pred,zero_division=0)),"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),"errors":int(np.sum(pred!=y))}
def routing(y,p,route):
    error=(p>=.5)!=y;hi=error&(np.maximum(p,1-p)>=.8);return {"routed":int(route.sum()),"captured_errors":int(np.sum(route&error)),"base_errors":int(error.sum()),"error_recall":float(np.sum(route&error)/max(error.sum(),1)),"routing_precision":float(np.sum(route&error)/max(route.sum(),1)),"highconf_errors":int(hi.sum()),"captured_highconf_errors":int(np.sum(route&hi)),"highconf_error_recall":float(np.sum(route&hi)/max(hi.sum(),1))}
def size_auc(x,size,seed):
    lo,hi=np.quantile(size,[.25,.75]);q=(size<=lo)|(size>=hi);z=(size[q]>=hi).astype(int);xx=x[q];pred=np.full(len(z),np.nan)
    for f,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(xx,z),1):
        m=Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,solver="liblinear",max_iter=5000,random_state=seed+f))]);m.fit(xx[tr],z[tr]);pred[va]=m.predict_proba(xx[va])[:,1]
    return float(roc_auc_score(z,pred)),int(len(z))

def main():
    cfg=json.loads((ROOT/"resolved_inputs.json").read_text());contract=json.loads((ROOT/"feature_contract.json").read_text());rows=read(ROOT/"lite_features_and_audit.csv");rr=read(cfg["router"]);rm={norm(r["sha256"]):r for r in rr}
    ids=np.asarray([norm(r["sha256"]) for r in rows]);names=contract["feature_names"];x=np.asarray([[float(r[n]) for n in names] for r in rows]);y=np.asarray([int(float(rm[s]["label"])) for s in ids]);folds=np.asarray([int(float(rm[s]["fold"])) for s in ids]);official=np.asarray([float(rm[s]["p_stage1"]) for s in ids]);official_route=np.asarray([int(float(rm[s]["route_DI"]))==1 for s in ids]);arms=arm_indices(names)
    if x.shape!=(799,82) or int(official_route.sum())!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: cohort/feature/route mismatch")
    probs={a:np.full((799,4),np.nan) for a in arms};fit_audit=[];timing=[]
    for f in sorted(set(folds)):
        tr=np.flatnonzero(folds!=f);va=np.flatnonzero(folds==f);seed=20260925+100000*f
        for arm,idx in arms.items():
            t=time.perf_counter();out,audit=committee(x[tr][:,idx],y[tr],x[va][:,idx],seed);probs[arm][va]=out;timing.append({"arm":arm,"fold":f,"dimension":len(idx),"seconds":time.perf_counter()-t});fit_audit.extend({"arm":arm,"fold":f,**r} for r in audit)
        print(f"LiteSemantic source OOF fold {f}/5 done",flush=True)
    aggregate=[];routes={};sample=[]
    aggregate.append({"arm":"FROZEN_GLOBAL24_DI25_REFERENCE",**classification(y,official),**routing(y,official,official_route),"shortcut_safe":"not_applicable"})
    for arm,idx in arms.items():
        p=probs[arm][:,0];decision=p>=.5;opp=np.where(decision[:,None],1-probs[arm],probs[arm]).mean(1);route=top_foldwise(opp,folds,ids);routes[arm]=(route,opp);aggregate.append({"arm":arm,"dimension":len(idx),**classification(y,p),**routing(y,p,route)})
    audit_cols=contract["audit_only_columns"];size=np.asarray([[float(r[n]) for n in audit_cols] for r in rows]);corr=[];shortcut=[]
    for arm,idx in arms.items():
        maxrho=0.0
        for local_i,global_i in enumerate(idx):
            for sj,sn in enumerate(audit_cols):
                rho=float(spearmanr(x[:,global_i],size[:,sj]).statistic);rho=0.0 if not np.isfinite(rho) else rho;maxrho=max(maxrho,abs(rho));corr.append({"arm":arm,"feature":names[global_i],"audit_size_variable":sn,"spearman_rho":rho,"abs_rho":abs(rho)})
        size_results=[]
        for sn in ("apk_bytes","method_ref_count","class_count"):
            auc,n=size_auc(x[:,idx],size[:,audit_cols.index(sn)],20260925);size_results.append(auc);shortcut.append({"arm":arm,"test":"predict_extreme_quartile__"+sn,"AUC":auc,"n":n})
        safe=maxrho<=.70 and max(size_results)<=.80
        for r in aggregate:
            if r["arm"]==arm:r["max_abs_size_spearman"]=maxrho;r["max_size_quartile_AUC"]=max(size_results);r["shortcut_safe"]=int(safe)
    for i,s in enumerate(ids):
        row={"sha256":s,"year":2017,"fold":int(folds[i]),"label":int(y[i]),"p_frozen_Global24":official[i],"route_frozen_DI25":int(official_route[i])}
        for arm in arms:row["p_stage1__"+arm]=probs[arm][i,0];row["route__"+arm]=int(routes[arm][0][i]);row["U_DI__"+arm]=routes[arm][1][i]
        sample.append(row)
    write(ROOT/"aggregate_metrics.csv",aggregate);write(ROOT/"feature_size_correlations.csv",corr);write(ROOT/"size_predictability.csv",shortcut);write(ROOT/"fit_audit.csv",fit_audit);write(ROOT/"timing_model.csv",timing);write(ROOT/"source_oof_predictions.csv",sample)
    extraction=np.asarray([float(r["total_seconds"]) for r in rows]);lines=["CARE-Droid strict-2017 LiteSemantic-NoShortcut source v1","","Manifest + DEX reference-table scan only; no Analysis object, call graph, Host, RoleGraph, or Ego.","Raw size/count fields are audit-only and never enter a model or router.","No package/certificate/time/SDK/year identity feature enters a model.","","Results:"]
    for r in aggregate:lines.append(f"{r['arm']}: BA={r['BA']:.6f} AUC={r['AUC']:.6f} F1={r['F1']:.6f} errors={r['errors']} captured={r['captured_errors']}/{r['base_errors']} recall25={r['error_recall']:.6f} precision25={r['routing_precision']:.6f} shortcut_safe={r['shortcut_safe']}")
    lines += ["",f"Raw extraction timing: total={extraction.sum():.3f}s mean={extraction.mean():.6f}s median={np.median(extraction):.6f}s p90={np.quantile(extraction,.9):.6f}s max={extraction.max():.6f}s", "Shortcut gate: max |Spearman| <= 0.70 and every extreme-size-quartile OOF AUC <= 0.80.","No arm is authorized for future evaluation or CARE-Droid replacement by this diagnostic alone."]
    (ROOT/"REPORT.txt").write_text("\n".join(lines)+"\n");print("LITESEMANTIC_NOSHORTCUT_SOURCE_DONE");print("\n".join(lines))

if __name__=="__main__":main()
