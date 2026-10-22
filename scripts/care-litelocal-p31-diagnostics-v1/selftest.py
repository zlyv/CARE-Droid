#!/usr/bin/env python3
import numpy as np
from p31_core import group_top,mean_opposition,mean_probability,label

def main():
    p=np.array([[.1,.2,.3,.4],[.9,.8,.7,.6],[.4,.6,.4,.6]])
    assert np.allclose(mean_probability(p),[.25,.75,.5])
    assert np.allclose(mean_opposition(p,mean_probability(p)),[.25,.25,.5])
    assert group_top(np.arange(8),np.repeat([2018,2019],4),np.array(list("abcdefgh")),.25).sum()==2
    assert label("benign")==0 and label("malicious")==1
    print("LITELOCAL_P31_DIAGNOSTICS_SELFTEST_PASS")
    print("candidate=P31_MEAN_COMMITTEE budget=0.25")
    print("callgraph_constructed=false stage2_training_invoked=false ego_consumed=false")

if __name__=="__main__":main()
