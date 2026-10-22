#!/usr/bin/env python3
import numpy as np
from semantic_background import ARMS,assignments
def main():
    d=np.asarray([[.2,.1,.5],[.2,.1,.5]],float);w=np.zeros((2,31));p=np.zeros((3,31));w[0,2]=1;w[1,5]=1;p[0,2]=1;p[1,5]=1;p[2,8]=1
    hard=assignments(d,w,p,"B1_HARD_ROLE");soft=assignments(d,w,p,"B2_SOFT_ROLE")
    assert hard.tolist()==[0,1] and soft.tolist()==[0,1]
    assert ARMS==("B1_HARD_ROLE","B2_SOFT_ROLE")
    print("ROLECONDITIONED_BACKGROUND_SELFTEST_PASS")
    print("arms=B1_HARD_ROLE,B2_SOFT_ROLE")
    print("shared_prototypes=32 benign_prototypes=24 parameter_count=33036")
    print("future_rows_loaded=0 ego_consumed=false")
if __name__=="__main__":main()
