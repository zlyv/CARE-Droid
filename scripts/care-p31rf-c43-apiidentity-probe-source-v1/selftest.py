#!/usr/bin/env python3
import numpy as np
from api_identity_core import reconstruct_hosts, FoldAPIProjector, jaccard

def payload():
    base=np.zeros((5,15),float);base[2:,4]=1
    return {"node_ids":["h0","h1","apiA","apiB","apiC"],
      "node_role_ids":[[],[],[1],[2],[1,3]],"node_base_features":base.tolist(),
      "edge_index":[[0,0,1],[2,3,4]],"edge_features":[[0]*7]*3,"edge_type_ids":["x"]*3}

def main():
    hosts=reconstruct_hosts(payload())
    assert len(hosts)==2 and hosts[0]["api_ids"]==("apiA","apiB") and hosts[1]["role_ids"]==(1,3)
    bags=[hosts,[hosts[0]],[hosts[1]],hosts]
    model=FoldAPIProjector(seed=7).fit(bags[:3]); out=model.transform(bags)
    assert out["role"].shape==(4,31) and out["api"].shape==(4,31)
    assert np.isfinite(out["api"]).all() and 0<=out["unknown_rate"].max()<=1
    assert jaccard(["a","b"],["b","c"])==1/3
    print("P31RF_C43_APIIDENTITY_SELFTEST_PASS")

if __name__=="__main__":main()

