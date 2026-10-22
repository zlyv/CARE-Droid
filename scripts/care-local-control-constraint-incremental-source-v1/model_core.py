#!/usr/bin/env python3
from __future__ import annotations
import hashlib,warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from feature_core import apply_role_mask

def fit_predict(x,y,test,config):
    if len(np.unique(y))!=2:raise RuntimeError('outer training has one class')
    model=make_pipeline(StandardScaler(),LogisticRegression(**{k:v for k,v in config.items() if k!='name'}))
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning);model.fit(x,y)
    return model.predict_proba(test)[:,1]

def matching_groups(train_p31,test_p31,train_base,test_base,train_hosts,test_hosts,train_cov,test_cov,seed):
    clusters=min(4,len(np.unique(train_p31,axis=0)))
    km=KMeans(n_clusters=max(clusters,1),n_init=10,random_state=seed).fit(train_p31)
    host_cut=float(np.median(train_hosts));unknown_cut=float(np.median(train_cov[:,-1]))
    def groups(p31,base,hosts,cov):
        return np.column_stack([km.predict(p31),np.asarray(base)>=.5,np.asarray(hosts)>host_cut,cov[:,-1]>unknown_cut]).astype(int)
    return groups(train_p31,train_base,train_hosts,train_cov),groups(test_p31,test_base,test_hosts,test_cov),dict(cluster_n=max(clusters,1),host_cut=host_cut,unknown_cut=unknown_cut)

def donor_permutation(groups,rng):
    donor=np.arange(len(groups));movable=np.zeros(len(groups),bool)
    for key in sorted(set(map(tuple,groups.tolist()))):
        ix=np.flatnonzero(np.all(groups==key,axis=1))
        if len(ix)>=2:
            # Random cyclic rotation within a shuffled group: no fixed donors.
            order=rng.permutation(ix);donor[order]=np.roll(order,1);movable[ix]=True
    return donor,movable

def conditional_shuffle(e,cov,groups,seed):
    donor,movable=donor_permutation(groups,np.random.default_rng(seed))
    out=apply_role_mask(e[donor],cov)
    return out,donor,movable,np.any(np.abs(out-e)>1e-12,axis=1)

def placebo(train_e,train_cov,test_cov,train_ids,test_ids,seed):
    # One frozen random realization per SHA; no labels or test moments enter.
    mean=train_e.mean(0);sd=train_e.std(0)
    def build(ids,cov):
        rows=[]
        for sid in ids:
            s=int.from_bytes(hashlib.sha256((str(seed)+':'+str(sid)).encode()).digest()[:8],'big')
            rows.append(np.clip(mean+sd*np.random.default_rng(s).standard_normal(124),0,1))
        return apply_role_mask(np.asarray(rows),cov)
    return build(train_ids,train_cov),build(test_ids,test_cov)
