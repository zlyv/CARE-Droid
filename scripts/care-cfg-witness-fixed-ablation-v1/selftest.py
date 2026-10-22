#!/usr/bin/env python3
"""Pure synthetic self-test for standardization and label-free conditional shuffle; writes no files."""
import numpy as np
from cfgwa_common import conditional_shuffle, standardize

x=np.arange(60,dtype=np.float32).reshape(20,3); size=np.column_stack([np.arange(20),np.arange(20)[::-1]]).astype(np.float32)
r=conditional_shuffle(x,size,size,np.random.RandomState(7))
assert r.shape==x.shape and sorted(map(tuple,r))==sorted(map(tuple,x))
a,b,m,s=standardize(x,r)
assert a.shape==b.shape==(20,3) and np.isfinite(a).all() and np.isfinite(b).all()
assert np.max(np.abs(a.mean(0)))<1e-5
print('CFG_WITNESS_ABLATION_SELFTEST_PASS')
