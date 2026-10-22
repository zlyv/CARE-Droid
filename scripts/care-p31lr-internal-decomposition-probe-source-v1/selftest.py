#!/usr/bin/env python3
import numpy as np
from decomposition_core import decompose,internal_knn_distance
def main():
    benign=decompose(np.array([.1,.2,.15]),np.array([.9,.8,.9]));isolated=decompose(np.array([.9,.2]),np.array([.05,.8]));pervasive=decompose(np.array([.9,.8,.85]),np.array([.9,.8,.9]))
    assert benign["score"]<0 and isolated["M_exception"]>.8 and pervasive["M_pervasive"]>.5
    x=np.array([[0.,0.],[0.1,0.],[3.,3.]]);w=np.eye(31)[[1,1,2]];a,_=internal_knn_distance(x,w,True,3)
    assert np.isfinite(a).all() and a[2]>a[0]
    print("P31LR_INTERNAL_DECOMPOSITION_SELFTEST_PASS")
    print("components=B_CONSENSUS,M_EXCEPTION,M_PERVASIVE protocols=ALL_HOST,ROLE_CONDITIONED,ROLE_SHARED_RELIABILITY")
    print("future_rows_loaded=0 neural_training_invoked=false")
if __name__=="__main__":main()
