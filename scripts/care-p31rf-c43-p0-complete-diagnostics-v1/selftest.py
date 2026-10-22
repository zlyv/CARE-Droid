#!/usr/bin/env python3
import numpy as np
from common import metric_counts,zero6
def main():
    bag=zero6([np.ones((2,105),np.float32)])[0]
    assert bag.shape==(2,111) and np.all(bag[:,105:]==0)
    c=metric_counts([1,0,1,0],[.2,.8,.8,.2],[.8,.2,.2,.8],[1,1,1,1])
    assert c["fixed"]==2 and c["broken"]==2 and c["fixed_FN"]==1 and c["fixed_FP"]==1
    print("P31RF_C43_P0_COMPLETE_SELFTEST_PASS")
    print("protocols=frozen_2017_to_future,independent_same_year_5fold")
    print("stage1=P31_RF router=RF_anchored committee_role=routing_only stage2=P0_ZERO6_C43_RQ")
if __name__=="__main__":main()
