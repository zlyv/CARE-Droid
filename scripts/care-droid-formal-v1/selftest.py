#!/usr/bin/env python3
import numpy as np
from common import metrics,deterministic_route
y=np.array([0,0,1,1]);base=np.array([.1,.8,.2,.9]);p=np.array([.1,.2,.8,.9]);route=np.array([0,1,1,0],bool)
m=metrics(y,base,p,route);assert m["fixed"]==2 and m["broken"]==0 and m["wrong"]==0
sha=np.array(["a","b","c","d"]);r1=deterministic_route(np.arange(4),sha,2,7);r2=deterministic_route(np.arange(4),sha,2,7);assert np.array_equal(r1,r2) and r1.sum()==2
print("CARE_DROID_FORMAL_V1_SELFTEST_PASS")

