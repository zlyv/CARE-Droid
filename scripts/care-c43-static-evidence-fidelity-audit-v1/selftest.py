#!/usr/bin/env python3
import numpy as np
from common import sketch_jaccard
def rank(values,x):
    values=np.asarray(values,float);return float((np.sum(values<x)+.5*np.sum(values==x))/len(values))
assert rank([1,2,3],2)==.5
def cosine(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);return float(1-np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))
assert abs(cosine([1,0],[1,0]))<1e-9
print("C43_STATIC_EVIDENCE_FIDELITY_SELFTEST_PASS")
