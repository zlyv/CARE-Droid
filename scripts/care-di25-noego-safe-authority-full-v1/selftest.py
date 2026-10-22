#!/usr/bin/env python3
import numpy as np
from safety_core import apply, choose

def main():
    y=np.array([1,1,0,0]);p=np.array([.2,.3,.2,.8]);q=np.array([.9,.8,.1,.6]);r=np.ones(4,bool)
    best,curve=choose(y,p,q,r); final,active,limited,delta=apply(p,q,r,best["threshold"])
    assert len(curve)==51 and final.shape==(4,) and np.max(np.abs(delta))<=2.25+1e-12
    print("SAFE_AUTHORITY_FULL_SELFTEST_PASS")
if __name__=="__main__":main()

