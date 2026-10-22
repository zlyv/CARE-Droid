#!/usr/bin/env python3
from __future__ import annotations

import hashlib, sys
from types import SimpleNamespace
import numpy as np

from common import FAMILY, SCRIPTS

WIDTH=16

def dependencies():
    if str(FAMILY) not in sys.path: sys.path.insert(0,str(FAMILY))
    from model_adapter import prepare_imports
    prepare_imports()
    import run_care_rolegraph_conditioned_dualstream_v4 as v4
    import haig_background as haig
    return v4,haig,v4.v3.v2,v4.v3.care_v1

def args():
    return SimpleNamespace(shared_prototypes=32,benign_prototypes=24,prototype_regions_per_apk=16,internal_background_coverage=.70)

def soft_assignment(distance, roles, profiles, scale, tau=1.0):
    d=np.asarray(distance,float)/np.maximum(np.asarray(scale,float).reshape(1,-1),1e-6)
    roles=np.asarray(roles,float); profiles=np.asarray(profiles,float)
    hp=np.argmax(roles,axis=1); pp=np.argmax(profiles,axis=1); hv=roles.sum(1)>0; pv=profiles.sum(1)>0
    logits=-.5*np.square(d/max(float(tau),1e-6))
    for i in range(len(roles)):
        mask=pv&(pp==hp[i]) if hv[i] else np.ones(len(profiles),bool)
        if np.any(mask): logits[i,~mask]=-np.inf
    maximum=np.max(logits,axis=1,keepdims=True); weight=np.exp(logits-maximum); weight[~np.isfinite(weight)]=0
    den=weight.sum(1,keepdims=True)
    return np.divide(weight,den,out=np.zeros_like(weight),where=den>0).astype(np.float32)

def scalar_summary(columns):
    x=np.asarray(columns,float); parts=[]
    for j in range(x.shape[1]):
        parts.extend([float(np.mean(x[:,j])),float(np.std(x[:,j])),float(np.quantile(x[:,j],.90)),float(np.max(x[:,j]))])
    return np.asarray(parts,np.float32)

def diagonal_state(bags,roles,centers,profiles,scales,haig,v2):
    dim=centers.shape[1]; k=len(centers); total=np.zeros((k,dim),float); count=np.zeros(k,float)
    global_values=np.concatenate(bags,axis=0); global_var=np.var(global_values,axis=0)+1e-3
    for x,w in zip(bags,roles):
        d=v2.pairwise_rms(x,centers); assignment=haig.hard(d/np.maximum(np.asarray(scales)[None,:],1e-6),w,profiles)
        for c in np.unique(assignment):
            z=x[assignment==c]-centers[int(c)]; total[int(c)]+=np.square(z).sum(0); count[int(c)]+=len(z)
    variance=np.empty_like(total)
    for c in range(k):
        local=total[c]/max(count[c],1.0); strength=count[c]/(count[c]+20.0); variance[c]=strength*local+(1-strength)*global_var
    mix=(count+1.0)/(count.sum()+k)
    return np.sqrt(np.maximum(variance,1e-4)).astype(np.float32),mix.astype(np.float32)

def moments(x,gamma,centers,std,mix):
    n=max(len(x),1); z=(x[:,None,:]-centers[None,:,:])/np.maximum(std[None,:,:],1e-4)
    first=np.sum(gamma[:,:,None]*z,axis=0)/(n*np.sqrt(np.maximum(mix[:,None],1e-6)))
    second=np.sum(gamma[:,:,None]*(np.square(z)-1.0),axis=0)/(n*np.sqrt(np.maximum(2*mix[:,None],1e-6)))
    return np.clip(first,-10,10).astype(np.float32),np.clip(second,-10,10).astype(np.float32)

def stable_rng(sid,seed):
    digest=hashlib.sha256((str(seed)+":"+str(sid)).encode()).digest(); return np.random.RandomState(int.from_bytes(digest[:4],"little"))

def permute_blocks(value,sid,seed):
    x=np.asarray(value); rng=stable_rng(sid,seed); order=rng.permutation(len(x)); return x[order]

def build_fold(train_bags,test_bags,labels,train_ids,test_ids,seed):
    v4,haig,v2,care_v1=dependencies(); cfg=args()
    raw_train,role_train,_=haig.split(train_bags); raw_test,role_test,_=haig.split(test_bags)
    mean,std=care_v1.fit_equal_apk_scaler(raw_train,np.arange(len(raw_train),dtype=np.int64))
    train=[((x-mean)/std).astype(np.float32) for x in raw_train]; test=[((x-mean)/std).astype(np.float32) for x in raw_test]
    state=v2.fit_prototype_state(train,np.asarray(labels,int),cfg,int(seed)); sp,bp=haig.profiles(train,role_train,labels,state,v2)
    shared_std,shared_mix=diagonal_state(train,role_train,state["shared_centers"],sp,state["shared_scale"],haig,v2)
    benign_scale=np.full(state["benign_k"],max(float(state["external_scale"]),1e-6),np.float32)
    benign_std,benign_mix=diagonal_state(train,role_train,state["benign_centers"],bp,benign_scale,haig,v2)

    def one(x,w,sid):
        sd=v2.pairwise_rms(x,state["shared_centers"]); bd=v2.pairwise_rms(x,state["benign_centers"])
        hard_s=haig.hard(sd/np.maximum(state["shared_scale"][None,:],1e-6),w,sp)
        hard_cols=haig.cols(x,w,state,sp,bp,cfg,v2); h0=scalar_summary(hard_cols)
        gs=soft_assignment(sd,w,sp,state["shared_scale"]); gb=soft_assignment(bd,w,bp,benign_scale)
        common=np.sum(gs*state["commonness"][None,:]*np.exp(-sd/np.maximum(state["shared_scale"][None,:],1e-6)),axis=1)
        external=np.sum(gb*np.log1p(bd/max(float(state["external_scale"]),1e-6)),axis=1)
        internal=v2.internal_dominant_distance(x,hard_s,cfg.internal_background_coverage)
        residual=(1-np.clip(common,0,1))*np.sqrt(np.maximum(external,0)*np.maximum(internal,0))
        b1=scalar_summary(np.stack([external,internal,common,residual],axis=1))
        fs,ss=moments(x,gs,state["shared_centers"],shared_std,shared_mix); fb,sb=moments(x,gb,state["benign_centers"],benign_std,benign_mix)
        first=np.concatenate([fs.reshape(-1),fb.reshape(-1)]); second=np.concatenate([ss.reshape(-1),sb.reshape(-1)])
        pfirst=np.concatenate([permute_blocks(fs,sid,seed).reshape(-1),permute_blocks(fb,sid,seed+1).reshape(-1)])
        psecond=np.concatenate([permute_blocks(ss,sid,seed+2).reshape(-1),permute_blocks(sb,sid,seed+3).reshape(-1)])
        nuisance=stable_rng(sid,seed+991).normal(size=WIDTH).astype(np.float32)
        return {
            "H0_HARD4":h0,"B1_SOFT4":b1,
            "B2_SIGNED_FIRST":np.concatenate([b1,first]),
            "B2U_UNSIGNED_FIRST":np.concatenate([b1,np.abs(first)]),
            "B3_SIGNED_FIRST_SECOND":np.concatenate([b1,first,second]),
            "B4_UNSIGNED_FIRST_SECOND":np.concatenate([b1,np.abs(first),second]),
            "B5_COMPONENT_SHUFFLE":np.concatenate([b1,pfirst,psecond]),
            "B6_CAPACITY_CONTROL":np.concatenate([b1,nuisance]),
        }
    train_rows=[one(x,w,sid) for x,w,sid in zip(train,role_train,train_ids)]
    test_rows=[one(x,w,sid) for x,w,sid in zip(test,role_test,test_ids)]
    arms=train_rows[0].keys()
    return ({a:np.asarray([r[a] for r in train_rows],np.float32) for a in arms},
            {a:np.asarray([r[a] for r in test_rows],np.float32) for a in arms},
            {"shared_k":int(state["shared_k"]),"benign_k":int(state["benign_k"]),"soft_temperature":1.0,
             "shared_profile_nonempty":int(np.sum(sp.sum(1)>0)),"benign_profile_nonempty":int(np.sum(bp.sum(1)>0)),
             "train_host_n":int(sum(map(len,train))),"test_host_n":int(sum(map(len,test)))})

def reduce(train,test,seed,width=WIDTH):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler(); a=scaler.fit_transform(np.asarray(train,float)); b=scaler.transform(np.asarray(test,float))
    n=min(int(width),a.shape[0]-1,a.shape[1]); pca=PCA(n_components=n,svd_solver="randomized",random_state=int(seed))
    at=pca.fit_transform(a); bt=pca.transform(b)
    if n<width:
        at=np.pad(at,((0,0),(0,width-n))); bt=np.pad(bt,((0,0),(0,width-n)))
    return at,bt,float(pca.explained_variance_ratio_.sum())
