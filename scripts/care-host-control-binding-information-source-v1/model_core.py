#!/usr/bin/env python3
from __future__ import annotations
import hashlib,warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def projection_map(width,seed):
    if width<=0:raise RuntimeError('invalid projection width')
    bucket=[];sign=[]
    for i in range(31*4*74):
        d=hashlib.sha256(f'{seed}:{i}'.encode()).digest()
        bucket.append(int.from_bytes(d[:8],'big')%width);sign.append(1 if d[8]&1 else -1)
    return np.asarray(bucket,int),np.asarray(sign,float)

def sketch(joint,bucket,sign,width):
    flat=np.asarray(joint,float).reshape(-1)
    if flat.shape!=(31*4*74,) or len(bucket)!=len(flat) or len(sign)!=len(flat) or not np.isfinite(flat).all():raise RuntimeError('joint sketch schema mismatch')
    return np.bincount(bucket,weights=flat*sign,minlength=width)

def variable_proof_mask(evidence):
    a=np.asarray(evidence,float)
    return np.any((a>0)&(a<1),axis=1)

def fit_predict(train,y,test,config):
    train=np.asarray(train,float);test=np.asarray(test,float);y=np.asarray(y)
    if train.ndim!=2 or test.ndim!=2 or train.shape[1]!=test.shape[1] or len(train)!=len(y) or len(np.unique(y))!=2 or not(np.isfinite(train).all() and np.isfinite(test).all()):raise RuntimeError('auxiliary train/test contract')
    model=make_pipeline(StandardScaler(),LogisticRegression(**{k:v for k,v in config.items() if k!='name'}))
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning);model.fit(train,y)
    return model.predict_proba(test)[:,1]

def placebo(train_binding,train_ids,test_ids,train_support,test_support,seed):
    mean=np.asarray(train_binding).mean(0);sd=np.asarray(train_binding).std(0)
    def build(ids,support):
        out=[]
        for sid,on in zip(ids,support):
            token=int.from_bytes(hashlib.sha256(f'{seed}:{sid}'.encode()).digest()[:8],'big')
            out.append((mean+sd*np.random.default_rng(token).standard_normal(len(mean))) if on else np.zeros_like(mean))
        return np.asarray(out)
    return build(train_ids,train_support),build(test_ids,test_support)
