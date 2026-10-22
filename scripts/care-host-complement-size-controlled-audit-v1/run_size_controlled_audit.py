#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_common import *

CORE=("A","B","C","ABC")
SIZE=("host_count","method_count","graph_node_count","graph_edge_count")

def graph_scores(prior_manifest, ids):
    rows=read_csv(Path(prior_manifest["strict_oof_probe_file"])); idc="sha256" if "sha256" in rows[0] else "sample_id"
    pc=next((x for x in ("probability","oof_probability","prob","p","score") if x in rows[0]),None)
    if pc is None: raise RuntimeError("BLOCKED_BY_CONTRACT: graph probe probability absent")
    out={}
    for r in rows:
        if r.get("task")=="need" and r.get("representation")=="B0_graph_token32":out[norm(r[idc])]=float(r[pc])
    if set(ids)-set(out):raise RuntimeError("BLOCKED_BY_CONTRACT: graph OOF scores incomplete")
    return np.asarray([out[s] for s in ids])

def fit_lr(x,y):
    return make_pipeline(StandardScaler(),LogisticRegression(penalty="l2",C=1.0,solver="liblinear",class_weight="balanced",max_iter=2000,random_state=SEED)).fit(x,y)

def main():
    manifest=load_json(ROOT/"preflight/source_manifest.json"); prior_manifest=load_json(PRIOR/"preflight/source_manifest.json")
    schema=load_json(PRIOR/"features/candidate_feature_schema.json")
    cohort={norm(r["sha256"]):r for r in read_csv(Path(manifest["cohort_file"]))}
    ids=sorted(s for s,r in cohort.items() if int(float(r["route_DI"]))==1)
    labels=np.asarray([int(float(cohort[s]["label"])) for s in ids]); stage1=np.asarray([float(cohort[s]["p_stage1"]) for s in ids])
    y=((stage1>=.5)!=labels).astype(int); years=np.asarray([int(float(cohort[s]["year"])) for s in ids]);folds=np.asarray([int(float(cohort[s]["fold"])) for s in ids])
    group_cols={"A":schema["candidate_groups"]["A_summary"],"B":schema["candidate_groups"]["B_summary"],"C":schema["candidate_groups"]["C_summary"]}
    group_cols["ABC"]=group_cols["A"]+group_cols["B"]+group_cols["C"]
    needed=set(SIZE)
    for cols in group_cols.values():needed.update(cols)
    features={}
    with (PRIOR/"features/candidate_features_summary.csv").open("r",encoding="utf-8",newline="") as f:
        for r in csv.DictReader(f):
            s=norm(r["sha256"])
            if s in cohort and int(float(cohort[s]["route_DI"]))==1:features[s]={k:float(r[k]) for k in needed}
    if set(features)!=set(ids):raise RuntimeError("BLOCKED_BY_CONTRACT: routed feature lineage mismatch")
    xs=np.asarray([[features[s][k] for k in SIZE] for s in ids],float)
    xc={name:np.asarray([[features[s][k] for k in cols] for s in ids],float) for name,cols in group_cols.items()}

    old_rows=read_csv(PRIOR/"predictions/candidate_oof_predictions.csv");old=defaultdict(dict)
    for r in old_rows:old[r["candidate"]][norm(r["sha256"])]=float(r["probability"])
    oldp={name:np.asarray([old[name][s] for s in ids]) for name in CORE}
    graph=graph_scores(prior_manifest,ids)

    variants=["SizeOnly"]+[f"{name}_PlusSize" for name in CORE]+[f"{name}_Residualized" for name in CORE]
    pred={name:np.full(len(ids),np.nan) for name in variants}; isolation=[]
    for year in YEARS:
        for fold in FOLDS:
            tr=np.flatnonzero((years==year)&(folds!=fold));va=np.flatnonzero((years==year)&(folds==fold))
            if set(tr)&set(va) or not len(va) or len(np.unique(y[tr]))!=2:raise RuntimeError("BLOCKED_BY_CONTRACT: fold isolation/target failure")
            pred["SizeOnly"][va]=fit_lr(xs[tr],y[tr]).predict_proba(xs[va])[:,1]
            for name in CORE:
                pred[f"{name}_PlusSize"][va]=fit_lr(np.c_[xs[tr],xc[name][tr]],y[tr]).predict_proba(np.c_[xs[va],xc[name][va]])[:,1]
                ss=StandardScaler().fit(xs[tr]);cs=StandardScaler().fit(xc[name][tr])
                zst,zsv=ss.transform(xs[tr]),ss.transform(xs[va]);zct,zcv=cs.transform(xc[name][tr]),cs.transform(xc[name][va])
                nuisance=LinearRegression().fit(zst,zct)
                rt=zct-nuisance.predict(zst);rv=zcv-nuisance.predict(zsv)
                pred[f"{name}_Residualized"][va]=fit_lr(rt,y[tr]).predict_proba(rv)[:,1]
            isolation.append({"year":year,"fold":fold,"train_n":len(tr),"validation_n":len(va),"overlap_n":0})
            print(f"size-controlled audit {year} fold={fold}/5 done",flush=True)
    if any(np.any(~np.isfinite(p)) for p in pred.values()):raise RuntimeError("OOF prediction incomplete")
    out=[]
    for i,s in enumerate(ids):
        base={"sha256":s,"year":years[i],"fold":folds[i],"need_correction":y[i],"graph_probability":graph[i],"SizeOnly":pred["SizeOnly"][i]}
        for name in CORE:base.update({f"{name}_Original":oldp[name][i],f"{name}_PlusSize":pred[f"{name}_PlusSize"][i],f"{name}_Residualized":pred[f"{name}_Residualized"][i]})
        out.append(base)
    write_csv(ROOT/"predictions/size_controlled_oof_predictions.csv",out);write_csv(ROOT/"audit/fold_isolation.csv",isolation)

    metric_rows=[]
    series={"SizeOnly":pred["SizeOnly"]}
    for name in CORE:series.update({f"{name}_Original":oldp[name],f"{name}_PlusSize":pred[f"{name}_PlusSize"],f"{name}_Residualized":pred[f"{name}_Residualized"]})
    for variant,p in series.items():
        for year in (*YEARS,"pooled"):
            m=np.ones(len(y),bool) if year=="pooled" else years==year
            metric_rows.append({"variant":variant,"year":year,**metrics(y[m],p[m])})
    write_csv(ROOT/"reports/metrics.csv",metric_rows)

    # Feature-level correlation, not candidate-score correlation.
    corr=[]
    for name in ("A","B","C"):
        for j,feature in enumerate(group_cols[name]):
            for k,size_name in enumerate(SIZE):
                rho=spearman(xc[name][:,j],xs[:,k]);corr.append({"candidate":name,"feature":feature,"size_variable":size_name,"rho":rho,"abs_rho":abs(rho)})
    write_csv(ROOT/"reports/feature_size_correlations.csv",sorted(corr,key=lambda r:r["abs_rho"],reverse=True))

    # Paired stratified bootstrap for the predeclared size-control questions.
    rng=np.random.default_rng(SEED);bootrows=[]
    pairs=[]
    for name in CORE:
        pairs += [("SizeOnly",f"{name}_PlusSize"),("Random",f"{name}_Residualized"),(f"{name}_Original",f"{name}_Residualized")]
    for baseline,candidate in pairs:
        pc=series[candidate];pb=None if baseline=="Random" else series[baseline]
        point_auc=auc(y,pc)-(.5 if pb is None else auc(y,pb));point_ap=ap(y,pc)-(float(np.mean(y)) if pb is None else ap(y,pb))
        ba=np.empty(BOOTSTRAPS);bp=np.empty(BOOTSTRAPS)
        for b in range(BOOTSTRAPS):
            idx=stratified_indices(y,years,rng);base_auc=.5 if pb is None else auc(y[idx],pb[idx]);base_ap=float(np.mean(y[idx])) if pb is None else ap(y[idx],pb[idx])
            ba[b]=auc(y[idx],pc[idx])-base_auc;bp[b]=ap(y[idx],pc[idx])-base_ap
        al,ah=ci(ba);pl,ph=ci(bp);bootrows.append({"baseline":baseline,"candidate":candidate,"delta_auc":point_auc,"delta_auc_ci_low":al,"delta_auc_ci_high":ah,"delta_pr_auc":point_ap,"delta_pr_auc_ci_low":pl,"delta_pr_auc_ci_high":ph,"iterations":BOOTSTRAPS,"seed":SEED})
    write_csv(ROOT/"reports/paired_bootstrap.csv",bootrows)

    # Graph-negative conditional separation for all controlled variants.
    cond=[];gneg=graph<.5
    for variant,p in series.items():
        err=gneg&(y==1);safe=gneg&(y==0);pos=p>=.5
        er=float(np.mean(pos[err]));fr=float(np.mean(pos[safe]));bb=np.empty(BOOTSTRAPS)
        ei=np.flatnonzero(err);si=np.flatnonzero(safe)
        for b in range(BOOTSTRAPS):
            e=[];s=[]
            for year in YEARS:
                ey=ei[years[ei]==year];sy=si[years[si]==year]
                if len(ey):e.extend(rng.choice(ey,len(ey),replace=True))
                if len(sy):s.extend(rng.choice(sy,len(sy),replace=True))
            bb[b]=np.mean(pos[e])-np.mean(pos[s])
        lo,hi=ci(bb);cond.append({"variant":variant,"error_n":int(sum(err)),"rescued":int(sum(pos&err)),"rescue_rate":er,"safe_n":int(sum(safe)),"false_rescue":int(sum(pos&safe)),"false_rescue_rate":fr,"conditional_separation":er-fr,"ci_low":lo,"ci_high":hi})
    write_csv(ROOT/"reports/conditional_separation.csv",cond)
    print("HOST_COMPLEMENT_SIZE_CONTROLLED_AUDIT_RUN_DONE")
if __name__=="__main__":main()
