#!/usr/bin/env python3
from __future__ import annotations
import math
import numpy as np

ROLE_N=31
MIN_CLASS_ROLE_SUPPORT=5
SHRINKAGE=0.20
ZERO_TOL=1e-12

def primary_roles(weights):
    w=np.asarray(weights,float); out=np.argmax(w,axis=1).astype(np.int16);out[w.sum(1)<=0]=-1;return out

def fit_role_models(feature_bags,role_bags,labels,min_support=MIN_CLASS_ROLE_SUPPORT,shrinkage=SHRINKAGE):
    """One APK-role centroid per class; no validation row is accepted here."""
    refs={(c,r):[] for c in (0,1) for r in range(ROLE_N)}
    for x,w,y in zip(feature_bags,role_bags,np.asarray(labels,int)):
        x=np.asarray(x,float);w=np.asarray(w,float)
        if len(x)!=len(w):raise RuntimeError("Host/role rows disagree")
        for r in np.flatnonzero(w.sum(0)>0):
            weight=w[:,r]; selected=weight>0
            refs[(int(y),int(r))].append(np.average(x[selected],axis=0,weights=weight[selected]))
    models={};audit=[]
    dim=feature_bags[0].shape[1]
    for r in range(ROLE_N):
        a=np.asarray(refs[(0,r)],float).reshape(-1,dim);b=np.asarray(refs[(1,r)],float).reshape(-1,dim)
        supported=len(a)>=min_support and len(b)>=min_support
        item={"supported":supported,"benign_n":len(a),"malware_n":len(b)}
        if supported:
            ma=a.mean(0);mb=b.mean(0);ss=((a-ma)**2).sum(0)+((b-mb)**2).sum(0);var=ss/max(len(a)+len(b)-2,1)
            target=max(float(np.median(var[var>1e-8])) if np.any(var>1e-8) else 1.0,1e-3)
            var=(1-shrinkage)*var+shrinkage*target;var=np.maximum(var,1e-4)
            if not np.isfinite(ma).all() or not np.isfinite(mb).all() or not np.isfinite(var).all():raise RuntimeError("nonfinite role model")
            item.update({"mu_ben":ma,"mu_mal":mb,"variance":var})
        models[r]=item;audit.append({"role":r,"benign_apk_role_n":len(a),"malware_apk_role_n":len(b),"supported":int(supported),"fallback":"ambiguous_excluded" if not supported else "none"})
    return models,audit

def host_signed_evidence(features,role_weights,models):
    x=np.asarray(features,float);w=np.maximum(np.asarray(role_weights,float),0)
    if x.ndim!=2 or w.shape!=(len(x),ROLE_N):raise RuntimeError("direction input shape mismatch")
    db=np.zeros(len(x));dm=np.zeros(len(x));total=np.zeros(len(x));support=np.zeros(len(x),int)
    for r,m in models.items():
        if not m["supported"]:continue
        weight=w[:,r];selected=weight>0
        if not np.any(selected):continue
        left=np.sqrt(np.mean((x[selected]-m["mu_ben"])**2/m["variance"],axis=1));right=np.sqrt(np.mean((x[selected]-m["mu_mal"])**2/m["variance"],axis=1))
        db[selected]+=weight[selected]*left;dm[selected]+=weight[selected]*right;total[selected]+=weight[selected];support[selected]+=1
    valid=total>0
    db=np.divide(db,total,out=np.zeros_like(db),where=valid);dm=np.divide(dm,total,out=np.zeros_like(dm),where=valid);signed=db-dm;signed[~valid]=0
    if not np.isfinite(signed).all():raise RuntimeError("nonfinite signed evidence")
    return db,dm,signed,support,valid

def aggregate(values,weights=None,mode="mean"):
    v=np.asarray(values,float)
    if not len(v):return 0.0
    if mode=="maxabs":return float(v[int(np.argmax(np.abs(v)))])
    if mode!="mean":raise KeyError(mode)
    if weights is None:return float(np.mean(v))
    q=np.maximum(np.asarray(weights,float),0);return float(np.sum(q*v)/q.sum()) if q.sum()>ZERO_TOL else float(np.mean(v))

def internal_current_loo(bag,assignment,coverage):
    """Current and LOO distances on the identical current-background scale."""
    # Match the frozen background implementation's float32 RMS arithmetic.
    x=np.asarray(bag,np.float32);a=np.asarray(assignment,int)
    if len(x)<=1:return np.zeros(len(x),float),np.zeros(len(x),float)
    counts=np.bincount(a);ordered=np.argsort(-counts);target=max(1,int(math.ceil(float(coverage)*len(x))));selected=[];covered=0
    for k in ordered:
        if counts[k]<=0:continue
        selected.append(int(k));covered+=int(counts[k])
        if covered>=target:break
    centers_current=np.asarray([x[a==k].mean(0) for k in selected],np.float32)
    raw_current=np.min(np.sqrt(np.maximum(np.mean(np.square(x[:,None,:]-centers_current[None,:,:]),axis=2),0.0)),axis=1)
    raw_loo=np.zeros(len(x),float)
    for i in range(len(x)):
        centers=[]
        for k in selected:
            idx=np.flatnonzero(a==k);idx=idx[idx!=i]
            if len(idx):centers.append(x[idx].mean(0))
        if not centers:
            idx=np.arange(len(x));idx=idx[idx!=i];centers=[x[idx].mean(0)]
        c=np.asarray(centers,np.float32);raw_loo[i]=np.min(np.sqrt(np.maximum(np.mean(np.square(c-x[i]),axis=1),0.0)))
    # In exact arithmetic, removing the evaluated Host cannot make its distance
    # to the frozen set of dominant-cluster centers smaller.  Float32 centroid
    # recomputation can violate that bound by a few ulps, so enforce the
    # mathematical lower bound before applying the shared scale.
    raw_loo=np.maximum(raw_loo,raw_current)
    positive=raw_current[raw_current>1e-8];scale=max(float(np.median(positive)) if len(positive) else 1.0,1e-6)
    current=np.log1p(raw_current/scale);loo=np.log1p(raw_loo/scale)
    return current,loo

def internal_loo(bag,assignment,coverage):
    return internal_current_loo(bag,assignment,coverage)[1]

def classify_direction(score,tol=ZERO_TOL):
    s=np.asarray(score,float);return np.where(s>tol,1,np.where(s<-tol,-1,0))
