#!/usr/bin/env python3
import numpy as np
from audit_collision import distances,fit_probe,probe_score,count_view
def main():
    x=np.asarray([[0.,0.],[0.,1.],[3.,3.],[3.,4.]]); y=np.asarray([0,0,1,1]); t=np.asarray([[.1,.2],[3.1,3.2]])
    for k in ("STANDARDIZED_EUCLIDEAN","COSINE","MAHALANOBIS"):
        d=distances(x,t,k); assert d.shape==(2,4) and np.isfinite(d).all()
    for name in ("LOGISTIC","LDA","CENTROID","KNN10"):
        s=probe_score(fit_probe(name,x,y),name,t); assert s.shape==(2,) and np.isfinite(s).all()
    assert count_view(np.asarray([1,2]),np.zeros((2,1546))).shape==(2,7)
    print("C43_REPRESENTATION_COLLISION_SELFTEST_PASS")
if __name__=="__main__":main()
