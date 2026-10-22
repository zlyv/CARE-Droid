#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from common import unit
MIN_SUPPORT=5

def make_apk_repr(bag):
    x=np.asarray(bag,float);core=unit(x[:,:43]);w=np.maximum(x[:,43:74],0);rv=np.zeros((31,43));mass=w.sum(0)
    for r in range(31):
        if mass[r]>1e-12:rv[r]=unit(np.sum(core*w[:,r:r+1],axis=0,keepdims=True))[0]
    valid=mass>1e-12
    gv=unit(np.sum(rv*mass[:,None],axis=0,keepdims=True))[0] if np.any(valid) else np.zeros(43)
    return rv,mass,gv

def prototypes(role,globalv,mass,y,indices,min_support=MIN_SUPPORT):
    idx=np.asarray(indices,int);y=np.asarray(y,int);pb=np.zeros((31,43));pm=np.zeros((31,43));valid=np.zeros(31,bool);sb=np.zeros(31,int);sm=np.zeros(31,int)
    for r in range(31):
        b=idx[(y[idx]==0)&(mass[idx,r]>0)];m=idx[(y[idx]==1)&(mass[idx,r]>0)];sb[r]=len(b);sm[r]=len(m)
        if len(b)>=min_support and len(m)>=min_support:pb[r]=unit(np.mean(role[b,r],axis=0,keepdims=True))[0];pm[r]=unit(np.mean(role[m,r],axis=0,keepdims=True))[0];valid[r]=True
    gb=unit(np.mean(globalv[idx[y[idx]==0]],axis=0,keepdims=True))[0];gm=unit(np.mean(globalv[idx[y[idx]==1]],axis=0,keepdims=True))[0]
    return {"pb":pb,"pm":pm,"valid":valid,"sb":sb,"sm":sm,"gb":gb,"gm":gm}

def scores(state,role,globalv,mass,indices):
    idx=np.asarray(indices,int);d1=globalv[idx]@state["gm"]-globalv[idx]@state["gb"];d2=np.zeros(len(idx));coverage=np.zeros(len(idx))
    per=np.full((len(idx),31),np.nan)
    for j,i in enumerate(idx):
        v=state["valid"]&(mass[i]>0)
        if np.any(v):
            delta=np.sum(role[i,v]*state["pm"][v],axis=1)-np.sum(role[i,v]*state["pb"][v],axis=1);ww=mass[i,v];d2[j]=np.sum(delta*ww)/np.sum(ww);coverage[j]=np.sum(ww)/max(np.sum(mass[i]),1e-12);per[j,np.flatnonzero(v)]=delta
    return d1,d2,coverage,per

def shuffled_roles(role,mass,indices,seed):
    rr=role.copy();mm=mass.copy();rng=np.random.RandomState(seed)
    for i in np.asarray(indices,int):
        p=rng.permutation(31);rr[i]=role[i,p];mm[i]=mass[i,p]
    return rr,mm

