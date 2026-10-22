#!/usr/bin/env python3
"""Fold-safe Compact43 backgrounds with H0-equivalent role distance weighting."""
from __future__ import annotations
import numpy as np

COMPACT=43; ROLE=31; RELATION=31; STORED=105
ROLE_SCALE=np.sqrt(86.0/74.0)
TAIL_SCALE=np.sqrt(43.0/74.0)


def split_bags(bags):
    compact=[]; roles=[]; relations=[]
    for bag in bags:
        x=np.asarray(bag,np.float32)
        if x.ndim!=2 or x.shape[1]!=STORED: raise RuntimeError(f"Compact sidecar bag must be N x 105, got {x.shape}")
        compact.append(x[:,:COMPACT]); weight=np.maximum(x[:,COMPACT:COMPACT+ROLE],0.0); den=weight.sum(axis=1,keepdims=True)
        roles.append(np.divide(weight,den,out=np.zeros_like(weight),where=den>0)); relations.append(x[:,COMPACT+ROLE:])
    return compact,roles,relations


def weighted_compact(bags,mean,std):
    result=[]
    for x in bags:
        z=(x-mean)/std
        result.append(np.concatenate([z[:,:31]*ROLE_SCALE,z[:,31:]*TAIL_SCALE],axis=1).astype(np.float32))
    return result


def prototype_profiles(feature_bags,role_bags,labels,state,v2):
    shared=np.zeros((state["shared_k"],ROLE),np.float64); benign=np.zeros((state["benign_k"],ROLE),np.float64)
    for x,w,y in zip(feature_bags,role_bags,np.asarray(labels,int)):
        if not len(x): continue
        sa=np.argmin(v2.pairwise_rms(x,state["shared_centers"]),axis=1)
        for k in np.unique(sa): shared[int(k)]+=w[sa==k].sum(axis=0)
        if y==0:
            ba=np.argmin(v2.pairwise_rms(x,state["benign_centers"]),axis=1)
            for k in np.unique(ba): benign[int(k)]+=w[ba==k].sum(axis=0)
    def normalize(a):
        den=a.sum(axis=1,keepdims=True); return np.divide(a,den,out=np.zeros_like(a),where=den>0).astype(np.float32)
    return normalize(shared),normalize(benign)


def hard_assignment(distance,weights,profiles):
    d=np.asarray(distance,float); w=np.asarray(weights,float); p=np.asarray(profiles,float)
    hp=np.argmax(w,axis=1); pp=np.argmax(p,axis=1); hv=w.sum(axis=1)>0; pv=p.sum(axis=1)>0; out=np.empty(len(w),np.int64)
    for i in range(len(w)):
        mask=pv&(pp==hp[i]) if hv[i] else np.zeros(len(p),bool)
        out[i]=int(np.argmin(np.where(mask,d[i],np.inf))) if np.any(mask) else int(np.argmin(d[i]))
    return out


def background_columns(bag,weights,state,sp,bp,args,v2):
    sd=v2.pairwise_rms(bag,state["shared_centers"]); sa=hard_assignment(sd/np.maximum(state["shared_scale"][None,:],1e-6),weights,sp)
    raw_shared=sd[np.arange(len(bag)),sa]; similarity=np.exp(-raw_shared/np.maximum(state["shared_scale"][sa],1e-6)); common=state["commonness"][sa]*similarity
    bd=v2.pairwise_rms(bag,state["benign_centers"]); ba=hard_assignment(bd,weights,bp); raw_external=bd[np.arange(len(bag)),ba]
    external=np.log1p(raw_external/max(float(state["external_scale"]),1e-6)).astype(np.float32); internal=v2.internal_dominant_distance(bag,sa,args.internal_background_coverage)
    residual=(1.0-np.clip(common,0.0,1.0))*np.sqrt(np.maximum(external,0.0)*np.maximum(internal,0.0))
    return np.stack([external,internal,common,residual],axis=1).astype(np.float32)


def make_fit(v3):
    v2=v3.v2
    def fit(train_bags,test_bags,labels,args,seed):
        raw_train,train_roles,train_rel=split_bags(train_bags); raw_test,test_roles,test_rel=split_bags(test_bags)
        mean,std=v3.care_v1.fit_equal_apk_scaler(raw_train,np.arange(len(raw_train),dtype=np.int64)); scaled_train=weighted_compact(raw_train,mean,std); scaled_test=weighted_compact(raw_test,mean,std)
        state=v2.fit_prototype_state(scaled_train,labels,args,int(seed)); sp,bp=prototype_profiles(scaled_train,train_roles,labels,state,v2)
        def augment(bags,roles,relations):
            output=[]; summaries=[]
            for x,w,r in zip(bags,roles,relations):
                columns=background_columns(x,w,state,sp,bp,args,v2); output.append(np.concatenate([x,columns,r],axis=1)); summaries.append([columns[:,0].mean(),columns[:,1].mean(),columns[:,2].mean(),columns[:,3].max()])
            return output,np.asarray(summaries,float)
        atr,ts=augment(scaled_train,train_roles,train_rel); ate,vs=augment(scaled_test,test_roles,test_rel)
        audit={"region_scaler_zero_std_n":int(np.sum(std==1.0)),"shared_prototype_n":int(state["shared_k"]),"benign_prototype_n":int(state["benign_k"]),"sampled_balanced_region_n":int(state["sampled_balanced_n"]),"sampled_benign_region_n":int(state["sampled_benign_n"]),"shared_commonness_mean":float(np.mean(state["commonness"])),"shared_commonness_max":float(np.max(state["commonness"])),"external_distance_scale":float(state["external_scale"]),"train_external_mean":float(ts[:,0].mean()),"test_external_mean":float(vs[:,0].mean()),"train_internal_mean":float(ts[:,1].mean()),"test_internal_mean":float(vs[:,1].mean()),"train_commonness_mean":float(ts[:,2].mean()),"test_commonness_mean":float(vs[:,2].mean()),"compact_host_width":43,"model_region_width":78,"duplicate_role_columns":0,"role_distance_weight":2.0,"shared_semantic_profile_nonempty":int(np.sum(sp.sum(axis=1)>0)),"benign_semantic_profile_nonempty":int(np.sum(bp.sum(axis=1)>0)),"prototype_capacity_matched":1}
        return atr,ate,audit
    return fit
