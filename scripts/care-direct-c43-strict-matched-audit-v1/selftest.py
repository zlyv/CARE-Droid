#!/usr/bin/env python3
import numpy as np
from interventions import *

x=np.arange(5*142,dtype=np.float32).reshape(5,142)
y=intervene_bag(x,"abc","N0_ROW_ORDER",0)
assert y.shape==x.shape
assert np.allclose(np.sort(y[:,0]),np.sort(x[:,0]))

base=np.asarray([.10,.12,.80,.82,.95])
host=np.asarray([10,11,20,22,200])
p31=np.zeros((5,31)); p31[0,0]=1; p31[1,0]=1; p31[2,1]=1; p31[3,1]=1; p31[4,2]=1
donor,diag=strict_matched_donor(base,host,p31,0)
assert donor.tolist()==[1,0,3,2,-1]
assert all(donor[i]<0 or (base[i]>=.5)==(base[donor[i]]>=.5) for i in range(5))
assert diag[-1] is None
print("DIRECT_C43_STRICT_MATCHED_SELFTEST_PASS arms=%d"%len(ARMS))
