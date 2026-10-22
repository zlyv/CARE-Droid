#!/usr/bin/env python3
import numpy as np
from common import binary_auc, match_pairs

def main():
    y=np.asarray([0,1,0,1,0,1]); pred=np.asarray([0,0,0,1,1,1],bool); x=np.asarray([[0],[.1],[.2],[1],[1.1],[1.2]],float)
    pairs=match_pairs(np.arange(6),y,pred,x,False,7)
    assert len(pairs)==2 and all(y[m]==1 and y[b]==0 and pred[m]==pred[b] for m,b,_,_ in pairs)
    assert abs(binary_auc([0,1,0,1],[0,1,.2,.8])-1.0)<1e-12
    print("C43_CONDITIONAL_MATCHED_SELFTEST_PASS unified_score=true same_side_matching=true",flush=True)
if __name__=="__main__": main()
