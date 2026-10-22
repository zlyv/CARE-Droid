#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from direction_core import (
    ROLE_N, aggregate, classify_direction, fit_role_models,
    host_signed_evidence, internal_current_loo,
)

def bag(center, role, n=3):
    x=np.tile(np.asarray(center,float),(n,1))
    x+=np.arange(n,dtype=float)[:,None]*0.01
    w=np.zeros((n,ROLE_N),float);w[:,role]=1.0
    return x,w

def main():
    feature=[];roles=[];labels=[]
    for y,center in ((0,[0.,0.,0.]),(1,[4.,4.,4.])):
        for k in range(6):
            x,w=bag(np.asarray(center)+k*.02,2)
            feature.append(x);roles.append(w);labels.append(y)
    models,audit=fit_role_models(feature,roles,np.asarray(labels))
    assert models[2]["supported"] and not models[3]["supported"]
    x,w=bag([3.8,4.1,4.0],2,2)
    db,dm,s,support,valid=host_signed_evidence(x,w,models)
    assert np.all(s>0) and np.all(db>dm) and np.all(valid) and np.all(support==1)
    x,w=bag([.1,.2,.0],2,2)
    db,dm,s,_,_=host_signed_evidence(x,w,models)
    assert np.all(s<0) and np.all(db<dm)
    x,w=bag([9.,9.,9.],3,2)
    db,dm,s,_,valid=host_signed_evidence(x,w,models)
    assert np.all(s==0) and not np.any(valid)
    assert aggregate([-2.,1.],mode="maxabs")==-2.
    assert aggregate([-2.,1.],[0.,3.])==1.
    assert np.array_equal(classify_direction([-1.,0.,1.]),[-1,0,1])
    current,loo=internal_current_loo(np.asarray([[0.,0.],[0.,0.],[3.,3.]]),np.asarray([0,0,1]),.70)
    assert loo.shape==(3,) and np.isfinite(loo).all() and np.all(loo>=current)
    assert {r["role"] for r in audit}==set(range(ROLE_N))
    print("P31LR_C43_ROLE_DIRECTION_PROBE_SELFTEST_PASS")
    print("signed_semantics=positive_malware negative_benign")
    print("unsupported_role_fallback=ambiguous_excluded")

if __name__=="__main__":main()
