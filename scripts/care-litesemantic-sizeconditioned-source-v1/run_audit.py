#!/usr/bin/env python3
"""Fold-safe size-nuisance residualization of locked LiteSemantic features."""
from __future__ import annotations
import csv,json,math,sys,time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import balanced_accuracy_score,confusion_matrix,f1_score,roc_auc_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer,StandardScaler

HERE=Path(__file__).resolve().parent;UPCODE=HERE.parent/"care-litesemantic-noshortcut-source-v1";sys.path.insert(0,str(UPCODE))
import run_source_oof as base
from feature_contract import arm_indices

UP=Path("artifacts/care-litesemantic-noshortcut-source-v1");OUT=Path("artifacts/care-litesemantic-sizeconditioned-source-v1")
NUISANCE=("apk_bytes","class_count","method_ref_count","string_count","permission_count","component_count","mapped_api_unique_count")

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
def nuisance_model():
    return Pipeline([("spline",SplineTransformer(n_knots=5,degree=2,knots="quantile",include_bias=False)),("scale",StandardScaler()),("ridge",Ridge(alpha=10.0))])
def residualize(xtr,xva,ztr,zva,seed):
    # Label-free nuisance cross-fitting: no malware label is available here.
    splitter=KFold(4,shuffle=True,random_state=seed);pred=np.full_like(xtr,np.nan,float)
    for fit,cal in splitter.split(ztr):
        m=nuisance_model();m.fit(ztr[fit],xtr[fit]);pred[cal]=m.predict(ztr[cal])
    if not np.isfinite(pred).all():raise RuntimeError("incomplete inner-OOF nuisance residuals")
    final=nuisance_model();final.fit(ztr,xtr)
    return xtr-pred,xva-final.predict(zva)
def exact(n01,n10):
    n=n01+n10
    if not n:return 1.0
    k=min(n01,n10);return min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/(2**n))
def metrics(y,p):
    pred=p>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();return {"BA":float(balanced_accuracy_score(y,pred)),"AUC":float(roc_auc_score(y,p)),"F1":float(f1_score(y,pred,zero_division=0)),"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),"errors":int(np.sum(pred!=y))}
def size_auc(x,size,seed):
    lo,hi=np.quantile(size,[.25,.75]);q=(size<=lo)|(size>=hi);z=(size[q]>=hi).astype(int);xx=x[q];p=np.full(len(z),np.nan)
    for f,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(xx,z),1):
        m=Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,solver="liblinear",max_iter=5000,random_state=seed+f))]);m.fit(xx[tr],z[tr]);p[va]=m.predict_proba(xx[va])[:,1]
    return float(roc_auc_score(z,p))

def main():
    OUT.mkdir(parents=True,exist_ok=True);contract=json.loads((UP/"feature_contract.json").read_text());resolved=json.loads((UP/"resolved_inputs.json").read_text());rows=read(UP/"lite_features_and_audit.csv");locked=read(UP/"source_oof_predictions.csv");rr=read(resolved["router"]);rm={norm(r["sha256"]):r for r in rr};lm={norm(r["sha256"]):r for r in locked}
    ids=np.asarray([norm(r["sha256"]) for r in rows]);names=contract["feature_names"];x=np.asarray([[float(r[n]) for n in names] for r in rows]);z=np.log1p(np.asarray([[float(r[n]) for n in NUISANCE] for r in rows]));y=np.asarray([int(float(rm[s]["label"])) for s in ids]);folds=np.asarray([int(float(rm[s]["fold"])) for s in ids]);indices=arm_indices(names)
    raw_p=np.asarray([float(lm[s]["p_stage1__L3_FUSED"]) for s in ids]);raw_route=np.asarray([int(float(lm[s]["route__L3_FUSED"]))==1 for s in ids])
    candidates={"RES_MANIFEST":indices["L0_MANIFEST"],"RES_ROLE":indices["L1_ROLE_PRESENCE"]+indices["L2_ROLE_PROPORTION"],"RES_FUSED":indices["L3_FUSED"]}
    probs={a:np.full((799,4),np.nan) for a in candidates};residuals={a:np.full((799,len(idx)),np.nan) for a,idx in candidates.items()};fit=[];timing=[]
    for fold in sorted(set(folds)):
        tr=np.flatnonzero(folds!=fold);va=np.flatnonzero(folds==fold);seed=20260925+fold*100000
        for arm,idx in candidates.items():
            t=time.perf_counter();rtr,rva=residualize(x[tr][:,idx],x[va][:,idx],z[tr],z[va],seed)
            # Only validation residuals form the final strict outer-OOF audit matrix.
            residuals[arm][va]=rva;out,audit=base.committee(rtr,y[tr],rva,seed);probs[arm][va]=out;timing.append({"arm":arm,"fold":fold,"seconds":time.perf_counter()-t});fit.extend({"arm":arm,"fold":fold,**r} for r in audit)
        print(f"LiteSemantic size-conditioned fold {fold}/5 done",flush=True)
    aggregate=[{"arm":"RAW_L3_LOCKED","dimension":82,**metrics(y,raw_p),**base.routing(y,raw_p,raw_route)}];routes={};corr=[];sizepred=[]
    for arm,idx in candidates.items():
        p=probs[arm][:,0];decision=p>=.5;score=np.where(decision[:,None],1-probs[arm],probs[arm]).mean(1);route=base.top_foldwise(score,folds,ids);routes[arm]=(route,score);maxrho=0
        for j,gi in enumerate(idx):
            for k,nuis in enumerate(NUISANCE):
                rho=float(spearmanr(residuals[arm][:,j],z[:,k]).statistic);rho=0 if not np.isfinite(rho) else rho;maxrho=max(maxrho,abs(rho));corr.append({"arm":arm,"feature":names[gi],"nuisance":nuis,"spearman_rho":rho,"abs_rho":abs(rho)})
        aucs=[]
        for k,nuis in enumerate(("apk_bytes","method_ref_count","class_count")):
            original=np.asarray([float(r[nuis]) for r in rows]);auc=size_auc(residuals[arm],original,20260925);aucs.append(auc);sizepred.append({"arm":arm,"test":"predict_extreme_quartile__"+nuis,"AUC":auc})
        aggregate.append({"arm":arm,"dimension":len(idx),**metrics(y,p),**base.routing(y,p,route),"max_abs_size_spearman":maxrho,"max_size_quartile_AUC":max(aucs),"shortcut_safe":int(maxrho<=.70 and max(aucs)<=.80)})
    paired=[]
    for arm in candidates:
        cp=probs[arm][:,0]>=.5;rp=raw_p>=.5;n01=int(np.sum((rp!=y)&(cp==y)));n10=int(np.sum((rp==y)&(cp!=y)));paired.append({"comparison":arm+"_minus_RAW_L3_LOCKED","candidate_only_correct":n01,"RAW_only_correct":n10,"paired_net":n01-n10,"exact_p":exact(n01,n10)})
    sample=[]
    for i,s in enumerate(ids):
        row={"sha256":s,"year":2017,"fold":int(folds[i]),"label":int(y[i]),"p__RAW_L3_LOCKED":raw_p[i],"route__RAW_L3_LOCKED":int(raw_route[i])}
        for arm in candidates:row["p__"+arm]=probs[arm][i,0];row["route__"+arm]=int(routes[arm][0][i]);row["U_DI__"+arm]=routes[arm][1][i]
        sample.append(row)
    write(OUT/"aggregate_metrics.csv",aggregate);write(OUT/"paired_vs_RAW_L3.csv",paired);write(OUT/"residual_size_correlations.csv",corr);write(OUT/"size_predictability.csv",sizepred);write(OUT/"fit_audit.csv",fit);write(OUT/"timing.csv",timing);write(OUT/"source_oof_predictions.csv",sample)
    lines=["CARE-Droid strict-2017 LiteSemantic size-conditioned source v1","","No APK reanalysis, future row, call graph, Stage-II training, or label-aware residualization.","Raw size variables are nuisance-only. Outer-train residuals use inner OOF nuisance fits; outer validation uses an outer-train-only nuisance fit.","Nuisance transform: quantile spline degree=2, five knots, Ridge alpha=10.","","Results:"]
    for r in aggregate:lines.append(f"{r['arm']}: BA={r['BA']:.6f} AUC={r['AUC']:.6f} F1={r['F1']:.6f} errors={r['errors']} captured={r['captured_errors']}/{r['base_errors']} recall25={r['error_recall']:.6f} shortcut_safe={r.get('shortcut_safe','reference')}")
    lines += ["","Paired versus locked raw L3:"]+[f"{r['comparison']}: candidate_only={r['candidate_only_correct']} RAW_only={r['RAW_only_correct']} net={r['paired_net']} p={r['exact_p']:.6f}" for r in paired]+["","No candidate is authorized for future evaluation until shortcut safety and source utility are inspected."]
    (OUT/"REPORT.txt").write_text("\n".join(lines)+"\n");print("LITESEMANTIC_SIZECONDITIONED_SOURCE_DONE");print("\n".join(lines))

if __name__=="__main__":main()
