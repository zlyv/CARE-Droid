#!/usr/bin/env python3
import numpy as np
from interventions import *

x=np.arange(5*142,dtype=np.float32).reshape(5,142)
for arm in ARMS[1:6]:
    y=intervene_bag(x,"abc",arm,0); assert y.shape==x.shape
assert np.allclose(np.sort(intervene_bag(x,"abc","I1_BACKGROUND_HOST_ALIGNMENT",0)[:,107]),np.sort(x[:,107]))
assert np.allclose(np.sort(intervene_bag(x,"abc","I2_RELATION_HOST_ALIGNMENT",0)[:,111]),np.sort(x[:,111]))
d=matched_donor([.1,.2,.8,.9],[1,2,3,4],0); assert sorted(d.tolist())==list(range(4)); assert all((np.asarray([.1,.2,.8,.9])>=.5)==(np.asarray([.1,.2,.8,.9])[d]>=.5))
print("DIRECT_C43_JOINT_AUDIT_SELFTEST_PASS arms=%d"%len(ARMS))

