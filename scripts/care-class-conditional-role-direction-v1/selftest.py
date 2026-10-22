#!/usr/bin/env python3
import numpy as np
from common import unit
x=unit([[3.,4.],[0.,0.]])
assert np.allclose(x[0],[.6,.8]) and np.allclose(x[1],0)
b=np.array([1.,0.]);m=np.array([0.,1.]);z=unit([[0.,2.]])[0]
score=float(z@m-z@b);assert score>0
print("CLASS_CONDITIONAL_ROLE_SELFTEST_PASS")

