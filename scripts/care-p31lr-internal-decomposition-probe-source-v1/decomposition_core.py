#!/usr/bin/env python3
from __future__ import annotations
import numpy as np

ROLE_N=31
EPS=1e-8

def primary_roles(weights):
    w=np.asarray(weights,float);out=np.argmax(w,axis=1).astype(np.int16)
    out[w.sum(1)<=0]=-1
    return out

def rms(a,b):
    return np.sqrt(np.maximum(np.mean((a-b)**2,axis=-1),0.0))

def fit_benign_role_models(bags,role_bags,labels,shrink_tau=10.0):
    refs={r:[] for r in range(ROLE_N)};all_refs=[]
    for x,w,y in zip(bags,role_bags,np.asarray(labels,int)):
        if y!=0:continue
        pr=primary_roles(w)
        for r in np.unique(pr[pr>=0]):
            z=np.mean(x[pr==r],axis=0);refs[int(r)].append(z);all_refs.append(z)
    if not all_refs:raise RuntimeError("BLOCKED_BY_CONTRACT: no benign references")
    all_refs=np.asarray(all_refs,float);gmu=all_refs.mean(0);gvar=np.maximum(all_refs.var(0),1e-3)
    models={};audit=[]
    for r in range(ROLE_N):
        a=np.asarray(refs[r],float).reshape(-1,all_refs.shape[1]);n=len(a);lam=n/(n+float(shrink_tau))
        if n:
            mu=lam*a.mean(0)+(1-lam)*gmu;var=lam*a.var(0)+(1-lam)*gvar
        else:mu=gmu.copy();var=gvar.copy()
        var=np.maximum(var,1e-3)
        models[r]={"mu":mu,"var":var,"n":n,"lambda":lam,"fallback":n==0}
        audit.append({"role":r,"benign_apk_role_n":n,"shrinkage_lambda":lam,"global_fallback":int(n==0)})
    models[-1]={"mu":gmu,"var":gvar,"n":len(all_refs),"lambda":0.,"fallback":True}
    return models,audit

def benign_distance(bag,roles,models):
    x=np.asarray(bag,float);pr=primary_roles(roles);out=np.zeros(len(x),float)
    for i,r in enumerate(pr):
        m=models[int(r) if int(r) in models else -1]
        out[i]=np.sqrt(np.mean((x[i]-m["mu"])**2/m["var"]))
    if not np.isfinite(out).all():raise RuntimeError("nonfinite benign distance")
    return out,pr

def internal_knn_distance(bag,roles,same_role,k=3):
    x=np.asarray(bag,float);pr=primary_roles(roles);n=len(x);out=np.full(n,np.inf)
    if n<=1:return out,pr
    dist=np.sqrt(np.maximum(np.mean((x[:,None,:]-x[None,:,:])**2,axis=2),0.0));np.fill_diagonal(dist,np.inf)
    for i in range(n):
        candidates=np.flatnonzero((pr==pr[i])&(np.arange(n)!=i)) if same_role and pr[i]>=0 else np.asarray([],int)
        if not len(candidates):candidates=np.flatnonzero(np.arange(n)!=i)
        values=np.sort(dist[i,candidates]);out[i]=float(np.median(values[:min(int(k),len(values))]))
    return out,pr

def empirical_cdf(value,reference):
    ref=np.sort(np.asarray(reference,float));v=np.asarray(value,float)
    if not len(ref):return np.full(len(v),.5,float)
    return (np.searchsorted(ref,v,side="right")+.5)/(len(ref)+1.)

def fit_percentile_references(bags,role_bags,labels,models,k=3):
    ext={r:[] for r in range(-1,ROLE_N)};ia={r:[] for r in range(-1,ROLE_N)};ir={r:[] for r in range(-1,ROLE_N)}
    for x,w,y in zip(bags,role_bags,np.asarray(labels,int)):
        if y!=0:continue
        d,pr=benign_distance(x,w,models);da,_=internal_knn_distance(x,w,False,k);dr,_=internal_knn_distance(x,w,True,k)
        for i,r in enumerate(pr):
            key=int(r);ext[key].append(float(d[i]));
            if np.isfinite(da[i]):ia[key].append(float(da[i]))
            if np.isfinite(dr[i]):ir[key].append(float(dr[i]))
            ext[-1].append(float(d[i]));
            if np.isfinite(da[i]):ia[-1].append(float(da[i]))
            if np.isfinite(dr[i]):ir[-1].append(float(dr[i]))
    return {"external":ext,"internal_all":ia,"internal_role":ir}

def host_axes(bag,roles,models,refs,k=3):
    d,pr=benign_distance(bag,roles,models);da,_=internal_knn_distance(bag,roles,False,k);dr,_=internal_knn_distance(bag,roles,True,k)
    q=np.zeros(len(d));ca=np.zeros(len(d));cr=np.zeros(len(d))
    for i,r in enumerate(pr):
        key=int(r);er=refs["external"].get(key) or refs["external"][-1]
        ar=refs["internal_all"].get(key) or refs["internal_all"][-1]
        rr=refs["internal_role"].get(key) or refs["internal_role"][-1]
        q[i]=empirical_cdf([d[i]],er)[0]
        ca[i]=0.0 if not np.isfinite(da[i]) else 1-empirical_cdf([da[i]],ar)[0]
        cr[i]=0.0 if not np.isfinite(dr[i]) else 1-empirical_cdf([dr[i]],rr)[0]
    return d,q,ca,cr,pr

def decompose(q,support,weight=None):
    q=np.clip(np.asarray(q,float),0,1);c=np.clip(np.asarray(support,float),0,1)
    w=np.ones(len(q),float) if weight is None else np.clip(np.asarray(weight,float),0,1)
    if not len(q) or w.sum()<=EPS:return {"B":0.,"M_exception":0.,"M_pervasive":0.,"score":0.}
    w=w/w.sum();ben=float(np.sum(w*c*(1-q)));per=float(np.sum(w*c*q));exc=float(np.max(q*(1-c)))
    # Fixed, label-free rule. Positive always means malware.
    score=.5*(exc+per)-ben
    return {"B":ben,"M_exception":exc,"M_pervasive":per,"score":score}
