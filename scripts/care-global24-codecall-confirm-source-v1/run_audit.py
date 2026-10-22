#!/usr/bin/env python3
"""Strict-2017 confirmation of the three Global24 Code/Call dimensions."""
from __future__ import annotations
import argparse,csv,json,math,sys,time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,confusion_matrix,f1_score,roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent;SCRIPTS=HERE.parent
for p in (HERE,SCRIPTS/"care-multiview-conformal-router-source-v1",SCRIPTS/"care-rolegraph-conditioned-v4-script",SCRIPTS/"care-rgcondattn-dedup-bounded-source-v1"):
    sys.path.insert(0,str(p))
import run_multiview_conformal_router_source_v1 as mv
from feature_contract import VARIANTS,retained_indices

OUT=Path("artifacts/care-global24-codecall-confirm-source-v1")


def norm(v):return str(v or "").strip().lower()


def read(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def write(path,rows):
    if not rows:return
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def logit(p):
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8);return np.log(p/(1-p))


def model(seed):
    return Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=1.0,penalty="l2",solver="liblinear",max_iter=5000,random_state=int(seed)))])


def calibrated_oof(xtr,ytr,xval,keep,seed,inner_folds):
    split=StratifiedKFold(n_splits=inner_folds,shuffle=True,random_state=seed)
    inner=np.full(len(ytr),np.nan)
    for k,(fit,cal) in enumerate(split.split(xtr,ytr),1):
        m=model(seed+100*k);m.fit(xtr[fit][:,keep],ytr[fit]);inner[cal]=m.predict_proba(xtr[cal][:,keep])[:,1]
    if not np.isfinite(inner).all():raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete inner OOF")
    calibrator=LogisticRegression(C=1e6,solver="lbfgs",max_iter=2000,random_state=seed+999)
    calibrator.fit(logit(inner).reshape(-1,1),ytr)
    m=model(seed+777);m.fit(xtr[:,keep],ytr)
    return np.clip(calibrator.predict_proba(logit(m.predict_proba(xval[:,keep])[:,1]).reshape(-1,1))[:,1],1e-8,1-1e-8)


def exact_p(n01,n10):
    n=int(n01+n10)
    if n==0:return 1.0
    k=min(int(n01),int(n10))
    return min(1.0,2.0*sum(math.comb(n,i) for i in range(k+1))/(2**n))


def metrics(y,p):
    pred=p>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"BA":float(balanced_accuracy_score(y,pred)),"AUC":float(roc_auc_score(y,p)),
            "F1":float(f1_score(y,pred,zero_division=0)),"TN":int(tn),"FP":int(fp),
            "FN":int(fn),"TP":int(tp),"errors":int(np.sum(pred!=y))}


def comparison(y,candidate,full):
    cp=candidate>=.5;fp=full>=.5
    n01=int(np.sum((fp!=y)&(cp==y)))
    n10=int(np.sum((fp==y)&(cp!=y)))
    return n01,n10,n01-n10,exact_p(n01,n10)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features",type=Path,default=Path("artifacts/p1-features/features.csv"))
    ap.add_argument("--schema",type=Path,default=Path("artifacts/p1-features/feature_schema.json"))
    ap.add_argument("--p72-oof",type=Path,default=Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"))
    ap.add_argument("--router",type=Path,default=Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv"))
    ap.add_argument("--inner-folds",type=int,default=4)
    ap.add_argument("--seed",type=int,default=20260925)
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    manifest,source_rows,_,compact_by_sha,compact_schema,original_names=mv.v4.audit.load_inputs(a)
    rr=read(a.router);router={norm(r["sha256"]):r for r in rr};ids=np.asarray([norm(r["sha256"]) for r in manifest])
    if len(manifest)!=799 or len(router)!=799 or set(ids)!=set(router) or {str(r.get("metadata_year","")) for r in source_rows}!={"2017"}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source-only loader mismatch")
    y=np.asarray([int(float(router[s]["label"])) for s in ids]);folds=np.asarray([int(float(router[s]["fold"])) for s in ids])
    official=np.asarray([float(router[s]["p_stage1"]) for s in ids]);route=np.asarray([int(float(router[s]["route_DI"]))==1 for s in ids])
    if int(route.sum())!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: frozen DI25 changed")
    probs={v:np.full(799,np.nan) for v in VARIANTS};timings=[];names_ref=None
    for fold in sorted(set(folds.tolist())):
        tr=np.flatnonzero(folds!=fold);va=np.flatnonzero(folds==fold)
        xtr,xva,names=mv.global_matrices([source_rows[i] for i in tr],[source_rows[i] for i in va],compact_by_sha,compact_schema,original_names)
        if names_ref is None:names_ref=list(names)
        elif list(names)!=names_ref:raise RuntimeError("BLOCKED_BY_CONTRACT: Global24 names changed across folds")
        # Capacity ablations must share the exact same inner folds and model
        # seeds.  A variant-specific seed would confound feature removal with
        # a different calibration split.
        protocol_seed=a.seed+fold*100000
        for variant in VARIANTS:
            keep=np.asarray(retained_indices(names_ref,variant),np.int64);t0=time.perf_counter()
            probs[variant][va]=calibrated_oof(xtr,y[tr],xva,keep,protocol_seed,a.inner_folds)
            timings.append({"variant":variant,"fold":fold,"protocol_seed":protocol_seed,
                            "retained_dimensions":len(keep),"seconds":time.perf_counter()-t0})
        print(f"Global24 Code/Call confirmation fold {fold}/5 done",flush=True)
    if any(not np.isfinite(p).all() for p in probs.values()):raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete source OOF")
    aggregate=[];per_fold=[];paired=[];stratified=[]
    for variant in VARIANTS:
        aggregate.append({"variant":variant,"retained_dimensions":len(retained_indices(names_ref,variant)),**metrics(y,probs[variant])})
        for fold in sorted(set(folds.tolist())):
            q=folds==fold;per_fold.append({"variant":variant,"fold":fold,**metrics(y[q],probs[variant][q])})
        for routed_name,mask in (("routed_DI25",route),("unrouted",~route)):
            stratified.append({"variant":variant,"stratum":routed_name,"n":int(mask.sum()),**metrics(y[mask],probs[variant][mask])})
    full=probs["FULL24"]
    for variant in VARIANTS[1:]:
        n01,n10,net,p=comparison(y,probs[variant],full)
        fold_d=[];fold_error=[]
        for fold in sorted(set(folds.tolist())):
            q=folds==fold;cm=metrics(y[q],probs[variant][q]);fm=metrics(y[q],full[q]);fold_d.append(cm["BA"]-fm["BA"]);fold_error.append(fm["errors"]-cm["errors"])
        paired.append({"comparison":variant+"_minus_FULL24","candidate_only_correct":n01,"FULL24_only_correct":n10,
                       "paired_net":net,"exact_p":p,"delta_BA":metrics(y,probs[variant])["BA"]-metrics(y,full)["BA"],
                       "delta_AUC":metrics(y,probs[variant])["AUC"]-metrics(y,full)["AUC"],
                       "positive_BA_folds":sum(v>1e-12 for v in fold_d),"nonnegative_BA_folds":sum(v>=-1e-12 for v in fold_d),
                       "fewer_error_folds":sum(v>0 for v in fold_error),"nonmore_error_folds":sum(v>=0 for v in fold_error)})
    sample=[]
    for i,sid in enumerate(ids):
        row={"sha256":sid,"year":2017,"fold":int(folds[i]),"label":int(y[i]),"p_stage1_frozen_official":float(official[i]),"route_DI_frozen":int(route[i])}
        for variant in VARIANTS:row["p__"+variant]=float(probs[variant][i]);row["pred__"+variant]=int(probs[variant][i]>=.5)
        sample.append(row)
    write(OUT/"aggregate_metrics.csv",aggregate);write(OUT/"per_fold_metrics.csv",per_fold)
    write(OUT/"paired_vs_FULL24.csv",paired);write(OUT/"route_stratified_metrics.csv",stratified)
    write(OUT/"source_oof_predictions.csv",sample);write(OUT/"timing.csv",timings)
    am={r["variant"]:r for r in aggregate};pm={r["comparison"]:r for r in paired}
    lines=["CARE-Droid strict-2017 Global24 Code/Call confirmation v1","",
           "Source-only five-fold OOF with identical folds, calibration, seeds, and classifier protocol.",
           "Frozen DI25 is identity-audited only: routed=200/799; it is not recomputed, blended, or replaced.",
           "No future row and no Stage-II training.","","Aggregate:"]
    for v in VARIANTS:
        r=am[v];lines.append(f"{v}: dims={r['retained_dimensions']} BA={r['BA']:.6f} AUC={r['AUC']:.6f} F1={r['F1']:.6f} errors={r['errors']}")
    lines += ["","Exact paired comparisons versus FULL24:"]
    for v in VARIANTS[1:]:
        r=pm[v+"_minus_FULL24"];lines.append(f"{v}: candidate_only={r['candidate_only_correct']} FULL24_only={r['FULL24_only_correct']} net={r['paired_net']} p={r['exact_p']:.6f} positive_BA_folds={r['positive_BA_folds']}/5 nonnegative_BA_folds={r['nonnegative_BA_folds']}/5")
    passing=[]
    for v in VARIANTS[1:]:
        r=pm[v+"_minus_FULL24"]
        if r["paired_net"]>0 and r["exact_p"]<=.05 and r["nonnegative_BA_folds"]>=4:passing.append(v)
    lines += ["",f"STRICT_SOURCE_SIMPLIFICATION_CANDIDATES={','.join(passing) if passing else 'NONE'}",
              "This source audit does not authorize a future diagnostic or replacement of the frozen official Stage-I.",
              "Individual-feature results diagnose which size variable drives the group effect; they do not alter DI25."]
    (OUT/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (OUT/"protocol.json").write_text(json.dumps({"source_only":True,"source_rows":799,"year":2017,"future_rows_loaded":0,
        "stage2_training_invoked":False,"frozen_DI25_modified":False,"route_DI":200,"outer_folds":5,
        "inner_calibration_folds":a.inner_folds,"variant_seed_policy":"same protocol seed for every variant within each outer fold",
        "variants":list(VARIANTS),"selection_rule":{"paired_net_gt_0":True,
        "exact_p_le":.05,"nonnegative_BA_folds_at_least":4},"strict_source_simplification_candidates":passing},indent=2)+"\n",encoding="utf-8")
    print("GLOBAL24_CODECALL_CONFIRM_SOURCE_DONE")
    print("\n".join(lines))


if __name__=="__main__":main()
