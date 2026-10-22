#!/usr/bin/env python3
"""Fold-safe backgrounds in the identical Compact43+Prov6 representation."""
from __future__ import annotations
import numpy as np

COMPACT=43; ROLE=31; RELATION=31; PROV=6; STORED=111
ROLE_SCALE=np.sqrt(98.0/80.0)
TAIL_PROV_SCALE=np.sqrt(49.0/80.0)

def split_bags(bags):
    features=[];roles=[];relations=[]
    for bag in bags:
        x=np.asarray(bag,np.float32)
        if x.ndim!=2 or x.shape[1]!=STORED:raise RuntimeError(f"Prov6 sidecar must be N x 111, got {x.shape}")
        feature=np.concatenate([x[:,:COMPACT],x[:,105:111]],axis=1)
        weight=np.maximum(x[:,COMPACT:COMPACT+ROLE],0);den=weight.sum(axis=1,keepdims=True)
        features.append(feature);roles.append(np.divide(weight,den,out=np.zeros_like(weight),where=den>0));relations.append(x[:,74:105])
    return features,roles,relations

def weighted(bags,mean,std):
    out=[]
    for x in bags:
        z=(x-mean)/std
        out.append(np.concatenate([z[:,:31]*ROLE_SCALE,z[:,31:]*TAIL_PROV_SCALE],axis=1).astype(np.float32))
    return out

def profiles(bags,roles,labels,state,v2):
    shared=np.zeros((state["shared_k"],ROLE));benign=np.zeros((state["benign_k"],ROLE))
    for x,w,y in zip(bags,roles,np.asarray(labels,int)):
        sa=np.argmin(v2.pairwise_rms(x,state["shared_centers"]),axis=1)
        for k in np.unique(sa):shared[int(k)]+=w[sa==k].sum(0)
        if y==0:
            ba=np.argmin(v2.pairwise_rms(x,state["benign_centers"]),axis=1)
            for k in np.unique(ba):benign[int(k)]+=w[ba==k].sum(0)
    def norm(a):
        d=a.sum(1,keepdims=True);return np.divide(a,d,out=np.zeros_like(a),where=d>0).astype(np.float32)
    return norm(shared),norm(benign)

def hard(distance,w,p):
    hp=np.argmax(w,1);pp=np.argmax(p,1);hv=w.sum(1)>0;pv=p.sum(1)>0;out=np.empty(len(w),np.int64)
    for i in range(len(w)):
        mask=pv&(pp==hp[i]) if hv[i] else np.zeros(len(p),bool)
        out[i]=int(np.argmin(np.where(mask,distance[i],np.inf))) if np.any(mask) else int(np.argmin(distance[i]))
    return out

def columns(x,w,state,sp,bp,args,v2):
    sd=v2.pairwise_rms(x,state["shared_centers"]);sa=hard(sd/np.maximum(state["shared_scale"][None,:],1e-6),w,sp);raw=sd[np.arange(len(x)),sa]
    common=state["commonness"][sa]*np.exp(-raw/np.maximum(state["shared_scale"][sa],1e-6))
    bd=v2.pairwise_rms(x,state["benign_centers"]);ba=hard(bd,w,bp);external=np.log1p(bd[np.arange(len(x)),ba]/max(float(state["external_scale"]),1e-6)).astype(np.float32)
    internal=v2.internal_dominant_distance(x,sa,args.internal_background_coverage);residual=(1-np.clip(common,0,1))*np.sqrt(np.maximum(external,0)*np.maximum(internal,0))
    return np.stack([external,internal,common,residual],1).astype(np.float32)

def make_fit(v3):
    v2=v3.v2
    def fit(train_bags,test_bags,labels,args,seed):
        rt,tr,rr=split_bags(train_bags);rv,tv,vr=split_bags(test_bags)
        mean,std=v3.care_v1.fit_equal_apk_scaler(rt,np.arange(len(rt),dtype=np.int64));st=weighted(rt,mean,std);sv=weighted(rv,mean,std)
        state=v2.fit_prototype_state(st,labels,args,int(seed));sp,bp=profiles(st,tr,labels,state,v2)
        def augment(bags,roles,relation):
            out=[];summary=[]
            for x,w,r in zip(bags,roles,relation):
                c=columns(x,w,state,sp,bp,args,v2);out.append(np.concatenate([x,c,r],1));summary.append([c[:,0].mean(),c[:,1].mean(),c[:,2].mean(),c[:,3].max()])
            return out,np.asarray(summary,float)
        atr,ts=augment(st,tr,rr);ate,vs=augment(sv,tv,vr)
        audit={"region_scaler_zero_std_n":int(np.sum(std==1)),"shared_prototype_n":int(state["shared_k"]),"benign_prototype_n":int(state["benign_k"]),"sampled_balanced_region_n":int(state["sampled_balanced_n"]),"sampled_benign_region_n":int(state["sampled_benign_n"]),"shared_commonness_mean":float(np.mean(state["commonness"])),"shared_commonness_max":float(np.max(state["commonness"])),"external_distance_scale":float(state["external_scale"]),"train_external_mean":float(ts[:,0].mean()),"test_external_mean":float(vs[:,0].mean()),"train_internal_mean":float(ts[:,1].mean()),"test_internal_mean":float(vs[:,1].mean()),"train_commonness_mean":float(ts[:,2].mean()),"test_commonness_mean":float(vs[:,2].mean()),"compact_host_width":49,"model_region_width":84,"provenance_width":6,"duplicate_role_columns":0,"role_distance_weight":2.0,"prototype_capacity_matched":1}
        return atr,ate,audit
    return fit
