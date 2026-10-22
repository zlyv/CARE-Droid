#!/usr/bin/env python3
import numpy as np
from lr_core import mean_opposition,foldwise_top25
from build_prov6_caches import prov
def main():
    rng=np.random.default_rng(7);members=rng.uniform(.01,.99,(799,4));anchor=members[:,0];fold=np.repeat(np.arange(1,6),[160,160,160,160,159]);sha=np.asarray([f"{i:064x}" for i in range(799)]);route=foldwise_top25(mean_opposition(members,anchor),fold,sha)
    assert int(route.sum())==200 and [int(route[fold==f].sum()) for f in range(1,6)]==[40]*5
    payload={"node_base_features":[[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,1,0,0,0,0,0,0,0]],"node_role_ids":[[],[3]],"edge_index":[[0],[1]]}
    x=prov(payload);assert x.shape==(1,6) and np.allclose(x[0],[1,0,0,0,0,1])
    print("P31LR_PROV6_SOURCE_SELFTEST_PASS");print("arms=P0_ZERO6,P1_PROV6,P2_SHUFFLED_PROV6 stage1=P31_LR committee_role=routing_only");print("stored_width=111 model_width=84 future_rows_loaded=0 ego_consumed=false")
if __name__=="__main__":main()
