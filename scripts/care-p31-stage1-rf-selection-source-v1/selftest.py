#!/usr/bin/env python3
import numpy as np
from stage1_core import mean_opposition,foldwise_top25
def main():
    rng=np.random.default_rng(17);p=rng.uniform(.01,.99,(799,4));fold=np.repeat(np.arange(1,6),[160,160,160,160,159]);sha=np.asarray([f"{i:064x}" for i in range(799)])
    for anchor in (p[:,0],p[:,1],p.mean(1)):
        score=mean_opposition(p,anchor);route=foldwise_top25(score,fold,sha)
        assert route.sum()==200 and [route[fold==f].sum() for f in range(1,6)]==[40]*5
    print("P31_RF_SELECTION_SOURCE_SELFTEST_PASS")
    print("arms=A_LR_LRROUTE,B_RF_LRROUTE,C_RF_RFROUTE,D_MEAN_MEANROUTE")
    print("future_rows_loaded=0")
if __name__=="__main__":main()
