#!/usr/bin/env python3
import numpy as np
from common import metrics

y=np.asarray([0,0,1,1]);b=np.asarray([.1,.8,.2,.9]);c=np.asarray([.1,.2,.8,.9])
m=metrics(y,b,c,np.ones(4,bool));assert (m["fixed"],m["broken"],m["net"])==(2,0,2)
print("SVM_CONSENSUS_DIAGNOSTICS_SELFTEST_PASS")
