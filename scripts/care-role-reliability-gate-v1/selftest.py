#!/usr/bin/env python3
import numpy as np
from common import entropy,logit
p=np.array([[1.,1.,0.],[0.,0.,0.]])
assert np.isfinite(entropy(p)).all() and abs(logit([.5])[0])<1e-10
base=np.array([0,1,0,1]);cand=np.array([1,0,0,1]);y=np.array([1,1,1,0])
fixed=(base!=y)&(cand==y);broken=(base==y)&(cand!=y)
assert fixed.tolist()==[True,False,False,False] and broken.tolist()==[False,True,False,False]
print("ROLE_RELIABILITY_GATE_SELFTEST_PASS")

