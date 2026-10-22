#!/usr/bin/env python3
import numpy as np
from analyze_mechanism import cohort, logit

assert [cohort(*x) for x in [(1,0,1),(0,0,0),(0,1,1),(0,1,0),(1,1,1),(1,1,0)]] == list("ABCDEF")
p=np.asarray([.1,.5,.9]); assert np.allclose(1/(1+np.exp(-logit(p))),p)
raw=np.asarray([-2.,0.,2.]); assert np.max(np.abs(2.25*np.tanh(raw)))<2.25
print("STAGE2_MECHANISM_SELFTEST_PASS")
