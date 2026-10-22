#!/usr/bin/env python3
from __future__ import annotations
import math
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from audit_common import *

REPS=("FUSED","HIDDEN")
PROBES=("LOGISTIC","LDA","CENTROID","KNN10")
SIDES=("FN_VS_TN","TP_VS_FP")

def sigmoid(x):
    x=np.clip(np.asarray(x,float),-40,40); return 1/(1+np.exp(-x))

def count_view(host,s7):
    s7=np.asarray(s7,float); parts=(s7[:,:120],s7[:,120:585],s7[:,585:])
    cols=[np.log1p(np.asarray(host,float))]
    for part in parts:
        cols += [np.log1p(np.count_nonzero(part,axis=1)),np.log1p(np.sum(np.abs(part),axis=1))]
    return np.column_stack(cols)

def fit_probe(name,x,y):
    if len(np.unique(y))<2:raise ValueError("one class")
    if name=="LOGISTIC":return make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",max_iter=3000,random_state=20261012)).fit(x,y)
    if name=="LDA":return make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto")).fit(x,y)
    if name=="CENTROID":return make_pipeline(StandardScaler(),NearestCentroid()).fit(x,y)
    if name=="KNN10":return make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=min(10,max(1,len(y)-1)),weights="distance")).fit(x,y)
    raise KeyError(name)

def probe_score(model,name,x):
    if hasattr(model,"predict_proba"):return np.asarray(model.predict_proba(x))[:,1]
    if hasattr(model,"decision_function"):return sigmoid(model.decision_function(x))
    # Pipeline + NearestCentroid exposes distances through the final estimator.
    scaled=model.steps[0][1].transform(x); centroid=model.steps[-1][1].centroids_
    dist=np.sqrt(np.maximum(((scaled[:,None,:]-centroid[None,:,:])**2).sum(2),0))
    return sigmoid(dist[:,0]-dist[:,1])

def distances(train,test,kind):
    train=np.asarray(train,float); test=np.asarray(test,float)
    if kind=="STANDARDIZED_EUCLIDEAN":
        scale=StandardScaler().fit(train); a=scale.transform(train); b=scale.transform(test)
        return np.sqrt(np.maximum(((b[:,None,:]-a[None,:,:])**2).sum(2),0))
    if kind=="COSINE":
        a=train/np.maximum(np.linalg.norm(train,axis=1,keepdims=True),1e-12); b=test/np.maximum(np.linalg.norm(test,axis=1,keepdims=True),1e-12)
        return 1-np.clip(b@a.T,-1,1)
    if kind=="MAHALANOBIS":
        scale=StandardScaler().fit(train); a=scale.transform(train); b=scale.transform(test); precision=LedoitWolf().fit(a).precision_
        delta=b[:,None,:]-a[None,:,:]; return np.sqrt(np.maximum(np.einsum("ntd,df,ntf->nt",delta,precision,delta),0))
    raise KeyError(kind)

def neighbor_rows(train_x,train_y,test_x,test_y,meta,rep,distance):
    d=distances(train_x,test_x,distance); rows=[]
    for i in range(len(test_y)):
        order=np.argsort(d[i],kind="mergesort"); same=np.flatnonzero(train_y==test_y[i]); opp=np.flatnonzero(train_y!=test_y[i])
        ds=float(np.min(d[i,same])) if len(same) else float("nan"); do=float(np.min(d[i,opp])) if len(opp) else float("nan")
        row={**meta[i],"representation":rep,"distance":distance,"d_same":ds,"d_opposite":do,"collision_margin":do-ds}
        for k in (5,10,20):row[f"purity_k{k}"]=float(np.mean(train_y[order[:min(k,len(order))]]==test_y[i]))
        rows.append(row)
    return rows

def strata(host):
    host=np.asarray(host,float); q=np.unique(np.quantile(host,[.25,.5,.75])); return np.digitize(host,q,right=True)

def permute_within(x,host,rng):
    out=np.asarray(x).copy(); bins=strata(host)
    for b in np.unique(bins):
        idx=np.flatnonzero(bins==b); out[idx]=out[rng.permutation(idx)]
    return out

def main():
    store=OUTPUT/"representation_store.npz"
    if not store.is_file():raise RuntimeError("BLOCKED_BY_CONTRACT: representation replay missing")
    ids,y,fold,base,route,expert,final,_=load_reference(False)
    with np.load(store,allow_pickle=False) as a:data={k:np.asarray(a[k]) for k in a.files}
    with np.load(P31,allow_pickle=False) as a:
        pids=[norm(x) for x in a["sha256"]]; pv=np.asarray(a["role_prevalence"],float)
    pp={sid:i for i,sid in enumerate(pids)}; p31=pv[[pp[s] for s in ids]]
    neighbor=[]; probe_fold=[]; pooled={}; prediction={name:base.copy() for name in [f"{rep}_{probe}" for rep in REPS for probe in PROBES]}
    fold_cache=[]
    for f in FOLDS:
        ti=np.asarray(data[f"f{f}_train_index"],int); vi=np.asarray(data[f"f{f}_test_index"],int)
        if not np.array_equal(ti,np.flatnonzero(fold!=f)) or not np.array_equal(vi,np.flatnonzero(fold==f)):raise RuntimeError("BLOCKED_BY_CONTRACT: stored index mismatch")
        train_views={"FUSED":data[f"f{f}_train_fused"],"HIDDEN":data[f"f{f}_train_hidden"],"P31":p31[ti],
            "COUNT":count_view(data[f"f{f}_train_host_count"],data[f"f{f}_train_s7"])}
        test_views={"FUSED":data[f"f{f}_test_fused"],"HIDDEN":data[f"f{f}_test_hidden"],"P31":p31[vi],
            "COUNT":count_view(data[f"f{f}_test_host_count"],data[f"f{f}_test_s7"])}
        train_base=np.asarray(data[f"f{f}_train_base"],float); test_base=np.asarray(data[f"f{f}_test_base"],float)
        train_y=np.asarray(data[f"f{f}_train_y"],int); test_y=y[vi]; test_route=route[vi]
        if not np.array_equal(train_y,y[ti]):raise RuntimeError("BLOCKED_BY_CONTRACT: train label alignment")
        fold_cache.append((f,ti,vi,train_views,test_views,train_base,test_base,train_y,test_y,data[f"f{f}_train_host_count"],data[f"f{f}_test_host_count"]))
        for side in SIDES:
            if side=="FN_VS_TN": tr=train_base<.5; te=(test_base<.5)&test_route
            else: tr=train_base>=.5; te=(test_base>=.5)&test_route
            probe_fold.append({"fold":f,"side":side,"representation":"SCALAR_DIRECT","probe":"SCALAR","n":int(te.sum()),"AUC":auc(test_y[te],expert[vi][te])})
            meta=[]
            for gi in vi[te]:
                before=(base[gi]>=.5)==y[gi]; after=(expert[gi]>=.5)==y[gi]
                category="FIXED" if (not before and after) else "BROKEN" if (before and not after) else "STABLE_CORRECT" if before else "PERSISTENT_ERROR"
                meta.append({"sha256":ids[gi],"fold":f,"side":side,"label":int(y[gi]),"category":category,"stage1_prob":float(base[gi]),"direct_prob":float(expert[gi])})
            for rep in ("FUSED","HIDDEN","P31","COUNT"):
                for dist in ("STANDARDIZED_EUCLIDEAN","COSINE","MAHALANOBIS"):
                    neighbor += neighbor_rows(train_views[rep][tr],train_y[tr],test_views[rep][te],test_y[te],meta,rep,dist)
                for probe in PROBES:
                    key=(rep,probe,side); model=fit_probe(probe,train_views[rep][tr],train_y[tr]); score=probe_score(model,probe,test_views[rep][te])
                    value=auc(test_y[te],score); probe_fold.append({"fold":f,"side":side,"representation":rep,"probe":probe,"n":int(te.sum()),"AUC":value})
                    pooled.setdefault(key,[]).append((test_y[te],score))
                    if rep in REPS: prediction[f"{rep}_{probe}"][vi[te]]=score
    # Pooled probe table and scalar reference.
    probe_pooled=[]
    for side in SIDES:
        mask=route&((base<.5) if side=="FN_VS_TN" else (base>=.5)); scalar=auc(y[mask],expert[mask])
        probe_pooled.append({"side":side,"representation":"SCALAR_DIRECT","probe":"SCALAR","n":int(mask.sum()),"AUC":scalar,"gain_vs_scalar":0.0})
        for rep in ("FUSED","HIDDEN","P31","COUNT"):
            for probe in PROBES:
                pieces=pooled[(rep,probe,side)]; yy=np.concatenate([p[0] for p in pieces]); ss=np.concatenate([p[1] for p in pieces]); value=auc(yy,ss)
                folds_better=sum(1 for r in probe_fold if r["side"]==side and r["representation"]==rep and r["probe"]==probe and math.isfinite(r["AUC"]) and r["AUC"]>next((q["AUC"] for q in probe_fold if q["side"]==side and q["fold"]==r["fold"] and q["representation"]=="P31" and q["probe"]=="LOGISTIC"),-1))
                probe_pooled.append({"side":side,"representation":rep,"probe":probe,"n":len(yy),"AUC":value,"gain_vs_scalar":value-scalar,"folds_better_than_p31_logistic":folds_better})
    # Matched shuffle null for the primary hidden cosine k=10 purity.
    rng=np.random.default_rng(20261012); null_rows=[]
    real_primary=[r for r in neighbor if r["representation"]=="HIDDEN" and r["distance"]=="COSINE"]
    for repeat in range(200):
        values=[]; broken_values=[]
        for f,ti,vi,trv,tev,trb,teb,try_,tey,trh,teh in fold_cache:
            for side in SIDES:
                tr=(trb<.5) if side=="FN_VS_TN" else (trb>=.5); te=(((teb<.5) if side=="FN_VS_TN" else (teb>=.5)) & route[vi])
                tx=permute_within(trv["HIDDEN"][tr],np.asarray(trh)[tr],rng); vx=permute_within(tev["HIDDEN"][te],np.asarray(teh)[te],rng)
                d=distances(tx,vx,"COSINE"); order=np.argsort(d,axis=1,kind="mergesort")[:,:min(10,d.shape[1])]; purity=np.mean(try_[tr][order]==tey[te,None],axis=1)
                values.extend(purity.tolist()); gis=vi[te]
                before=(base[gis]>=.5)==y[gis]; after=(expert[gis]>=.5)==y[gis]; broken=before&(~after); broken_values.extend(purity[broken].tolist())
        null_rows.append({"repeat":repeat,"mean_purity_k10":float(np.mean(values)),"broken_mean_purity_k10":float(np.mean(broken_values)) if broken_values else float("nan")})
    real_all=float(np.mean([r["purity_k10"] for r in real_primary])); real_broken=float(np.mean([r["purity_k10"] for r in real_primary if r["category"]=="BROKEN"]))
    null95=float(np.quantile([r["broken_mean_purity_k10"] for r in null_rows],.95))
    # Descriptive direct-decision results from every frozen probe; never deployment selection.
    integration=[]
    for name,score in prediction.items():
        final_probe=base.copy(); final_probe[route]=score[route]; integration.append({"arm":name,**metrics(y,base,final_probe,route)})
    # Frozen decision: require corroboration from at least two simple C43 probes on both sides.
    scalar={r["side"]:r["AUC"] for r in probe_pooled if r["representation"]=="SCALAR_DIRECT"}
    null_best={side:max(r["AUC"] for r in probe_pooled if r["side"]==side and r["representation"] in ("P31","COUNT")) for side in SIDES}
    primary=[]
    for rep in REPS:
        for probe in ("LOGISTIC","LDA"):
            rows={r["side"]:r for r in probe_pooled if r["representation"]==rep and r["probe"]==probe}
            fold_ok=True
            for side in SIDES:
                candidate=[r for r in probe_fold if r["side"]==side and r["representation"]==rep and r["probe"]==probe and math.isfinite(r["AUC"])]
                scalar_fold={r["fold"]:r["AUC"] for r in probe_fold if r["side"]==side and r["representation"]=="SCALAR_DIRECT" and math.isfinite(r["AUC"])}
                comparable=[r for r in candidate if r["fold"] in scalar_fold]; needed=max(3,int(math.ceil(.75*len(comparable))))
                if len(comparable)<3 or sum(r["AUC"]>scalar_fold[r["fold"]] for r in comparable)<needed:fold_ok=False
            if fold_ok and all(rows[s]["AUC"]>=scalar[s]+.03 and rows[s]["AUC"]>=null_best[s]+.03 for s in SIDES):primary.append(rep+"_"+probe)
    readout= len(primary)>=2 and real_broken>null95
    collision=(not readout) and len(primary)==0 and real_broken<=null95
    status="DIRECT_C43_READOUT_BOTTLENECK_SUPPORTED" if readout else "C43_REPRESENTATION_COLLISION_SUPPORTED" if collision else "C43_REPRESENTATION_COLLISION_AUDIT_INCONCLUSIVE"
    write_csv(OUTPUT/"neighbor_samples.csv",neighbor); write_csv(OUTPUT/"probe_fold_results.csv",probe_fold); write_csv(OUTPUT/"probe_pooled_results.csv",probe_pooled)
    write_csv(OUTPUT/"matched_null_results.csv",null_rows); write_csv(OUTPUT/"probe_direct_decision_diagnostics.csv",integration)
    dump_json(OUTPUT/"collision_audit.json",{"status":status,"scalar_auc":scalar,"best_P31_or_count_auc":null_best,"primary_passing_probes":primary,
        "hidden_cosine_purity_k10":real_all,"broken_hidden_cosine_purity_k10":real_broken,"matched_null_broken_purity_95pct":null95,
        "criteria":{"at_least_two_primary_probes_pass_both_sides":len(primary)>=2,"broken_local_purity>matched_null95":real_broken>null95},"future_rows_loaded":0})
    print("C43_REPRESENTATION_COLLISION_AUDIT_DONE status="+status,flush=True)
if __name__=="__main__":main()
