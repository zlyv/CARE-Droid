#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
COMPACT=107;ROLE=31;REL=31;STORED=169
def split(bags):
    f=[];r=[];q=[]
    for b in bags:
        x=np.asarray(b,np.float32)
        if x.ndim!=2 or x.shape[1]!=STORED:raise RuntimeError(f"HAIG Host bag must be N x {STORED}, got {x.shape}")
        f.append(np.concatenate([x[:,:43],x[:,105:]],1));w=np.maximum(x[:,43:74],0);d=w.sum(1,keepdims=True);r.append(np.divide(w,d,out=np.zeros_like(w),where=d>0));q.append(x[:,74:105])
    return f,r,q
def profiles(bags,roles,labels,state,v2):
    s=np.zeros((state["shared_k"],ROLE));b=np.zeros((state["benign_k"],ROLE))
    for x,w,y in zip(bags,roles,np.asarray(labels,int)):
        sa=np.argmin(v2.pairwise_rms(x,state["shared_centers"]),1)
        for k in np.unique(sa):s[int(k)]+=w[sa==k].sum(0)
        if y==0:
            ba=np.argmin(v2.pairwise_rms(x,state["benign_centers"]),1)
            for k in np.unique(ba):b[int(k)]+=w[ba==k].sum(0)
    def n(a):d=a.sum(1,keepdims=True);return np.divide(a,d,out=np.zeros_like(a),where=d>0).astype(np.float32)
    return n(s),n(b)
def hard(d,w,p):
    hp=np.argmax(w,1);pp=np.argmax(p,1);hv=w.sum(1)>0;pv=p.sum(1)>0;o=np.empty(len(w),int)
    for i in range(len(w)):
        m=pv&(pp==hp[i]) if hv[i] else np.zeros(len(p),bool);o[i]=int(np.argmin(np.where(m,d[i],np.inf))) if np.any(m) else int(np.argmin(d[i]))
    return o
def cols(x,w,state,sp,bp,args,v2):
    sd=v2.pairwise_rms(x,state["shared_centers"]);sa=hard(sd/np.maximum(state["shared_scale"][None,:],1e-6),w,sp);raw=sd[np.arange(len(x)),sa];common=state["commonness"][sa]*np.exp(-raw/np.maximum(state["shared_scale"][sa],1e-6));bd=v2.pairwise_rms(x,state["benign_centers"]);ba=hard(bd,w,bp);ext=np.log1p(bd[np.arange(len(x)),ba]/max(float(state["external_scale"]),1e-6));internal=v2.internal_dominant_distance(x,sa,args.internal_background_coverage);res=(1-np.clip(common,0,1))*np.sqrt(np.maximum(ext,0)*np.maximum(internal,0));return np.stack([ext,internal,common,res],1).astype(np.float32)
def make_fit(v3):
    v2=v3.v2
    def fit(train,test,labels,args,seed):
        rt,tr,rr=split(train);rv,tv,vr=split(test);mean,std=v3.care_v1.fit_equal_apk_scaler(rt,np.arange(len(rt)));st=[((x-mean)/std).astype(np.float32) for x in rt];sv=[((x-mean)/std).astype(np.float32) for x in rv];state=v2.fit_prototype_state(st,labels,args,int(seed));sp,bp=profiles(st,tr,labels,state,v2)
        def aug(xs,ws,rels):
            out=[];summ=[]
            for x,w,r in zip(xs,ws,rels):c=cols(x,w,state,sp,bp,args,v2);out.append(np.concatenate([x,c,r],1));summ.append([c[:,0].mean(),c[:,1].mean(),c[:,2].mean(),c[:,3].max()])
            return out,np.asarray(summ)
        a,ts=aug(st,tr,rr);b,vs=aug(sv,tv,vr);return a,b,{"compact_host_width":COMPACT,"model_region_width":COMPACT+4+REL,"prototype_capacity_matched":1,"train_external_mean":float(ts[:,0].mean()),"test_external_mean":float(vs[:,0].mean())}
    return fit

