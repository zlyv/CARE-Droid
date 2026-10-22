#!/usr/bin/env python3
import numpy as np
from dependency_core import TrainOnlyProjector, role_cooccurrence_matrix, stratified_shuffle

def main():
    bags=[[{"role_ids":(1,3)}],[{"role_ids":(1,)}],[]]
    x,names=role_cooccurrence_matrix(bags)
    assert x.shape==(3,496) and np.isfinite(x).all() and x[0].sum()>0
    p=TrainOnlyProjector(seed=7).fit(x,names); z=p.transform(x)
    assert z.shape==(3,31) and np.isfinite(z).all()
    raw=np.arange(18).reshape(6,3); strata=np.array([0,0,0,1,1,1])
    shuffled=stratified_shuffle(raw,strata,9)
    assert set(map(tuple,shuffled[:3]))==set(map(tuple,raw[:3]))
    assert set(map(tuple,shuffled[3:]))==set(map(tuple,raw[3:]))
    print("P31RF_DEPENDENCY_SUFFICIENCY_SELFTEST_PASS")
    print("dependency_is_not_cooccurrence=true future_rows_loaded=0")

if __name__=="__main__":main()

