#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from common import K,MIN_SUPPORT,unit

def make_representations(host_values,offsets):
    n=len(offsets)-1;role=np.zeros((n,31,43));mass=np.zeros((n,31));apk=np.zeros((n,43));host_count=np.diff(offsets).astype(int)
    for i in range(n):
        bag=np.asarray(host_values[offsets[i]:offsets[i+1]],float)
        if bag.ndim!=2 or bag.shape[1]<74:raise RuntimeError(f"invalid Host bag {i}: {bag.shape}")
        core=unit(bag[:,:43]);weight=np.maximum(bag[:,43:74],0);mass[i]=weight.sum(0)
        for r in range(31):
            if mass[i,r]>1e-12:role[i,r]=unit(np.sum(core*weight[:,r:r+1],axis=0,keepdims=True))[0]
        valid=mass[i]>1e-12
        if np.any(valid):apk[i]=unit(np.sum(role[i,valid]*mass[i,valid,None],axis=0,keepdims=True))[0]
    if not np.isfinite(role).all() or not np.isfinite(apk).all():raise RuntimeError("NaN/Inf representation")
    return role,mass,apk,host_count

def _top(query,bank,index,k=K,exclude_ge=None,rng=None):
    index=np.asarray(index,int)
    if len(index)==0:return float("nan"),[],[]
    sim=np.asarray(bank[index]@query,float)
    keep=np.ones(len(index),bool) if exclude_ge is None else sim<exclude_ge
    index=index[keep];sim=sim[keep]
    if len(index)==0:return float("nan"),[],[]
    take=min(k,len(index))
    if rng is None:
        order=np.lexsort((index,-sim))[:take]
    else:
        order=rng.choice(len(index),size=take,replace=False)
    return float(np.mean(sim[order])),index[order].tolist(),sim[order].tolist()

def apk_scores(apk,y,train,test,k=K,labels=None,random_neighbors=False,exclude_ge=None,seed=0,collect=False):
    labels=np.asarray(y if labels is None else labels,int);train=np.asarray(train,int);rng=np.random.RandomState(seed) if random_neighbors else None
    out=np.full(len(y),np.nan);detail=[];neighbors=[]
    for i in np.asarray(test,int):
        b=train[labels[train]==0];m=train[labels[train]==1]
        sb,bi,bs=_top(apk[i],apk,b,k,exclude_ge,rng);sm,mi,ms=_top(apk[i],apk,m,k,exclude_ge,rng)
        if np.isfinite(sb) and np.isfinite(sm):out[i]=sm-sb
        detail.append({"index":i,"benign_similarity":sb,"malware_similarity":sm,"direction_score":out[i],"benign_neighbors":len(bi),"malware_neighbors":len(mi)})
        if collect:
            for cls,ids,sims in ((0,bi,bs),(1,mi,ms)):
                for rank,(j,s) in enumerate(zip(ids,sims),1):neighbors.append((i,-1,cls,rank,j,s))
    return out,detail,neighbors

def role_scores(role,mass,y,train,test,k=K,labels=None,random_neighbors=False,exclude_ge=None,seed=0,collect=False,min_support=MIN_SUPPORT):
    labels=np.asarray(y if labels is None else labels,int);train=np.asarray(train,int);rng=np.random.RandomState(seed) if random_neighbors else None
    out=np.full(len(y),np.nan);detail=[];neighbors=[];support=[]
    banks={}
    for r in range(31):
        b=train[(labels[train]==0)&(mass[train,r]>0)];m=train[(labels[train]==1)&(mass[train,r]>0)];valid=len(b)>=min_support and len(m)>=min_support
        banks[r]=(b,m,valid);support.append((r,len(b),len(m),valid))
    for i in np.asarray(test,int):
        values=[];weights=[];valid_roles=0;bn=0;mn=0
        for r in range(31):
            b,m,valid=banks[r]
            if not valid or mass[i,r]<=0:continue
            sb,bi,bs=_top(role[i,r],role[:,r],b,k,exclude_ge,rng);sm,mi,ms=_top(role[i,r],role[:,r],m,k,exclude_ge,rng)
            if not np.isfinite(sb) or not np.isfinite(sm):continue
            values.append(sm-sb);weights.append(mass[i,r]);valid_roles+=1;bn+=len(bi);mn+=len(mi)
            if collect:
                for cls,ids,sims in ((0,bi,bs),(1,mi,ms)):
                    for rank,(j,s) in enumerate(zip(ids,sims),1):neighbors.append((i,r,cls,rank,j,s))
        if values:out[i]=float(np.average(values,weights=np.asarray(weights,float)))
        detail.append({"index":i,"direction_score":out[i],"valid_query_roles":valid_roles,"role_mass_coverage":float(np.sum(weights)/max(np.sum(mass[i]),1e-12)),"benign_neighbors":bn,"malware_neighbors":mn})
    return out,detail,neighbors,support

def shuffled_roles(role,mass,indices,seed):
    rr=role.copy();mm=mass.copy();rng=np.random.RandomState(seed)
    for i in np.asarray(indices,int):
        perm=rng.permutation(31);rr[i]=role[i,perm];mm[i]=mass[i,perm]
    return rr,mm

