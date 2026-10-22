#!/usr/bin/env python3
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import compact_background
from role_api_core import Registry,bag_tokens,project

REGISTRY=None; ARM=None; PROJECTION_AUDITS=[]; APK_FEATURE_AUDITS=[]; VOCABULARIES={}; CACHE={}
def configure(path,arm):
    global REGISTRY,ARM; REGISTRY=Registry(path); ARM=arm
def _key(bags):
    values=[]
    for bag in bags:
        x=np.asarray(bag); values.extend(int(round(float(v))) for v in x[:,-1])
    return hashlib.sha256(np.asarray(values,np.int64).tobytes()).hexdigest()
def make_fit(v3):
    base_fit=compact_background.make_fit(v3)
    def fit(train,test,labels,args,seed):
        if REGISTRY is None or ARM is None: raise RuntimeError("role API registry not configured")
        base_train=[np.asarray(x,np.float32)[:,:105] for x in train]; base_test=[np.asarray(x,np.float32)[:,:105] for x in test]
        aug_train,aug_test,audit=base_fit(base_train,base_test,labels,args,seed)
        key=(ARM,_key(train),_key(test)); cached=CACHE.get(key)
        if cached is None:
            tt=bag_tokens(train,REGISTRY); tv=bag_tokens(test,REGISTRY); shuffle=ARM in ("R2S_FIXED_SHUFFLED","R4_SHUFFLED_WITHIN_ROLE")
            bt,bv,pa=project(tt,tv,shuffle,abs(hash(key))%(2**31-1)); CACHE[key]=(bt,bv,pa); cached=CACHE[key]
        bt,bv,pa=cached; public={k:v for k,v in pa.items() if not k.startswith("_")};VOCABULARIES.setdefault(pa["vocabulary_sha256"],{"tokens":pa["_tokens"],"frequency":pa["_frequency"]})
        def attach(base,blocks):
            out=[]
            for x,b in zip(base,blocks):
                flat=np.asarray(b,np.float32).reshape(len(x),-1)
                if flat.shape!=(len(x),31*16): raise RuntimeError("role API block shape mismatch")
                out.append(np.concatenate([x,flat],axis=1))
            return out
        train_out,test_out=attach(aug_train,bt),attach(aug_test,bv)
        full=len(train) in (639,640)
        PROJECTION_AUDITS.append({"arm":ARM,"seed":int(seed),"train_apk_n":len(train),"test_apk_n":len(test),"full_outer_fit":int(full),**public})
        if full:
            for bag,block in zip(test,bv):
                apk_indices={int(REGISTRY.host_apk[int(round(float(row[-1])))]) for row in np.asarray(bag) if int(round(float(row[-1])))>=0}
                if len(apk_indices)==1:
                    APK_FEATURE_AUDITS.append({"arm":ARM,"apk_index":next(iter(apk_indices)),"api_residual_input_norm":float(np.linalg.norm(block)/max(np.sqrt(max(len(block),1)),1.0)),"api_present_role_n":int(np.sum(np.linalg.norm(block,axis=2).sum(axis=0)>0))})
        audit.update(public); audit.update({"api_dim":16,"role_conditioned_width":496,"api_residual_enters_background":0})
        return train_out,test_out,audit
    return fit
