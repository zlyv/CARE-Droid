#!/usr/bin/env python3
import numpy as np
from tb_frs import soft_assignment,scalar_summary,moments

def main():
    d=np.asarray([[.1,1.,2.],[1.,.2,2.]])
    w=np.asarray([[1.,0.],[0.,1.]])
    p=np.asarray([[1.,0.],[0.,1.],[1.,0.]])
    g=soft_assignment(d,w,p,np.ones(3))
    assert np.allclose(g.sum(1),1) and g[0,1]==0 and g[1,0]==0 and g[1,2]==0
    s=scalar_summary(np.arange(20,dtype=float).reshape(5,4)); assert s.shape==(16,)
    x=np.asarray([[1.,-1.],[2.,-2.]])
    c=np.zeros((1,2)); std=np.ones((1,2)); mix=np.ones(1); gamma=np.ones((2,1))
    first,second=moments(x,gamma,c,std,mix); assert first[0,0]>0 and first[0,1]<0 and not np.allclose(np.abs(first),first)
    print("TBFRS_SELFTEST_PASS soft_role_mask=true signed_direction=true width=16",flush=True)
if __name__=="__main__":main()
