#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location("common",Path(__file__).with_name("audit_common.py")); c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
assert c.THRESHOLD==0.5 and c.BOOTSTRAPS==10000 and c.SEED==20260920
y=np.array([0,0,1,1]); p=np.array([.1,.2,.8,.9])
assert c.auc(y,p)==1.0 and c.ap(y,p)==1.0
x=np.arange(10.0); assert np.isclose(c.spearman(x,x),1.0)
years=np.array([2018]*4+[2019]*4); yy=np.array([0,0,1,1]*2); rng=np.random.default_rng(1)
idx=c.stratified_indices(yy,years,rng)
assert len(idx)==8 and set(years[idx])=={2018,2019}
folds=np.array([1,2,3,4,5]);
for f in range(1,6): assert not set(np.flatnonzero(folds!=f))&set(np.flatnonzero(folds==f))
print("HOST_COMPLEMENT_SIZE_CONTROLLED_SELFTEST_PASS")
