#!/usr/bin/env python3
"""Capacity-matched semantic conditioning of the frozen three backgrounds."""
from __future__ import annotations
import numpy as np

FEATURE_DIM=74
ROLE_DIM=31
ARMS=("B1_HARD_ROLE","B2_SOFT_ROLE")

def split_bags(bags):
    features=[];roles=[]
    for bag in bags:
        x=np.asarray(bag,np.float32)
        if x.ndim!=2 or x.shape[1]!=(FEATURE_DIM+ROLE_DIM):
            raise RuntimeError(f"semantic sidecar bag must be N x 105, got {x.shape}")
        features.append(x[:,:FEATURE_DIM]);w=np.maximum(x[:,FEATURE_DIM:],0.0)
        den=w.sum(axis=1,keepdims=True);roles.append(np.divide(w,den,out=np.zeros_like(w),where=den>0))
    return features,roles

def prototype_profiles(feature_bags,role_bags,labels,state,v2):
    shared=np.zeros((state["shared_k"],ROLE_DIM),np.float64)
    benign=np.zeros((state["benign_k"],ROLE_DIM),np.float64)
    for x,w,y in zip(feature_bags,role_bags,np.asarray(labels,int)):
        if not len(x):continue
        sa=np.argmin(v2.pairwise_rms(x,state["shared_centers"]),axis=1)
        for k in np.unique(sa):shared[int(k)]+=w[sa==k].sum(axis=0)
        if y==0:
            ba=np.argmin(v2.pairwise_rms(x,state["benign_centers"]),axis=1)
            for k in np.unique(ba):benign[int(k)]+=w[ba==k].sum(axis=0)
    def normalize(a):
        den=a.sum(axis=1,keepdims=True)
        return np.divide(a,den,out=np.zeros_like(a),where=den>0).astype(np.float32)
    return normalize(shared),normalize(benign)

def assignments(distance,weights,profiles,mode,scale=None):
    d=np.asarray(distance,np.float64)
    w=np.asarray(weights,np.float64);p=np.asarray(profiles,np.float64)
    if scale is not None:d=d/np.maximum(np.asarray(scale,np.float64)[None,:],1e-6)
    if mode=="B2_SOFT_ROLE":
        compatibility=np.clip(w@p.T,0.0,1.0)
        score=d+(1.0-compatibility)
        empty=w.sum(axis=1)<=0
        if np.any(empty):score[empty]=d[empty]
        return np.argmin(score,axis=1)
    if mode!="B1_HARD_ROLE":raise KeyError(mode)
    host_primary=np.argmax(w,axis=1);proto_primary=np.argmax(p,axis=1)
    host_valid=w.sum(axis=1)>0;proto_valid=p.sum(axis=1)>0
    out=np.empty(len(w),np.int64)
    for i in range(len(w)):
        mask=proto_valid&(proto_primary==host_primary[i]) if host_valid[i] else np.zeros(len(p),bool)
        out[i]=int(np.argmin(np.where(mask,d[i],np.inf))) if np.any(mask) else int(np.argmin(d[i]))
    return out

def semantic_columns(bag,weights,state,shared_profile,benign_profile,args,v2,mode):
    bag=np.asarray(bag,np.float32);weights=np.asarray(weights,np.float32)
    sd=v2.pairwise_rms(bag,state["shared_centers"])
    sa=assignments(sd,weights,shared_profile,mode,state["shared_scale"])
    raw_shared=sd[np.arange(len(bag)),sa]
    similarity=np.exp(-raw_shared/np.maximum(state["shared_scale"][sa],1e-6))
    common=state["commonness"][sa]*similarity
    bd=v2.pairwise_rms(bag,state["benign_centers"])
    ba=assignments(bd,weights,benign_profile,mode)
    raw_external=bd[np.arange(len(bag)),ba]
    external=np.log1p(raw_external/max(float(state["external_scale"]),1e-6)).astype(np.float32)
    internal=v2.internal_dominant_distance(bag,sa,args.internal_background_coverage)
    residual=(1.0-np.clip(common,0.0,1.0))*np.sqrt(np.maximum(external,0.0)*np.maximum(internal,0.0))
    return np.stack([external,internal,common,residual],axis=1).astype(np.float32)

def make_fit(v3,mode):
    if mode not in ARMS:raise KeyError(mode)
    v2=v3.v2
    def fit(train_bags,test_bags,labels,args,seed):
        raw_train,train_roles=split_bags(train_bags);raw_test,test_roles=split_bags(test_bags)
        mean,std=v3.care_v1.fit_equal_apk_scaler(raw_train,np.arange(len(raw_train),dtype=np.int64))
        scaled_train=[(x-mean)/std for x in raw_train];scaled_test=[(x-mean)/std for x in raw_test]
        state=v2.fit_prototype_state(scaled_train,labels,args,int(seed))
        sp,bp=prototype_profiles(scaled_train,train_roles,labels,state,v2)
        def augment(bags,roles):
            result=[];summaries=[]
            for x,w in zip(bags,roles):
                columns=semantic_columns(x,w,state,sp,bp,args,v2,mode)
                result.append(np.concatenate([x,columns],axis=1));summaries.append([float(columns[:,0].mean()),float(columns[:,1].mean()),float(columns[:,2].mean()),float(columns[:,3].max())])
            return result,np.asarray(summaries,np.float64)
        atr,ts=augment(scaled_train,train_roles);ate,vs=augment(scaled_test,test_roles)
        audit={"region_scaler_zero_std_n":int(np.sum(std==1.0)),"shared_prototype_n":int(state["shared_k"]),"benign_prototype_n":int(state["benign_k"]),"sampled_balanced_region_n":int(state["sampled_balanced_n"]),"sampled_benign_region_n":int(state["sampled_benign_n"]),"shared_commonness_mean":float(np.mean(state["commonness"])),"shared_commonness_max":float(np.max(state["commonness"])),"external_distance_scale":float(state["external_scale"]),"train_external_mean":float(ts[:,0].mean()),"test_external_mean":float(vs[:,0].mean()),"train_internal_mean":float(ts[:,1].mean()),"test_internal_mean":float(vs[:,1].mean()),"train_commonness_mean":float(ts[:,2].mean()),"test_commonness_mean":float(vs[:,2].mean()),"semantic_background_mode":mode,"shared_semantic_profile_nonempty":int(np.sum(sp.sum(axis=1)>0)),"benign_semantic_profile_nonempty":int(np.sum(bp.sum(axis=1)>0)),"prototype_capacity_matched":1}
        return atr,ate,audit
    return fit
