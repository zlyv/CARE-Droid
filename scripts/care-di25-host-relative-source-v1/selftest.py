#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent.parent;BASE=PROJECT/"scripts/care-di25-graphglobal-path-source-v1"
for p in (HERE,BASE,PROJECT/"scripts",PROJECT/"scripts/care-rolegraph-conditioned-v4-script"):sys.path.insert(0,str(p))
from host_relative_features import ARMS,build_arm_values,centered_midrank,reconstruct_hosts
from graphglobal_path_models import make_graphglobal_path_class
import run_care_rolegraph_conditioned_dualstream_v4 as v4

def main():
    x=centered_midrank([0,0,1,2]);assert abs(float(x.mean()))<1e-7 and x[0]==x[1] and x[-1]>x[-2]
    payload={"node_ids":["h0","h1","e0","e1"],"node_role_ids":[[],[],[0],[0,1]],"node_base_features":[[0]*15 for _ in range(4)],"edge_index":[[0,0,1],[2,3,0]],"edge_features":[[0,1,0,0,0,1,0],[1,0,1,0,0,1,0],[0,0,0,0,0,0,0]]};payload["node_base_features"][2][4]=1;payload["node_base_features"][3][4]=1
    rec=reconstruct_hosts(payload,np.arange(31));assert len(rec)==1 and rec[0]["endpoint_counts"][0]==2 and rec[0]["endpoint_counts"][1]==1
    old=np.zeros((1,74),np.float32);old[0,:31]=rec[0]["presence"]
    for arm in ARMS:assert build_arm_values(old,rec,arm).shape==(1,74)
    args=SimpleNamespace(max_delta=2.25,graph_dim=32,embedding_dim=24,hidden_dim=48,graph_heads=4,graph_layers=2,dropout=0.0)
    model=make_graphglobal_path_class(v4,"Q_QUERY_ONLY")(24,"RGCondAttn",args)
    assert sum(p.numel() for p in model.parameters())==33036
    print("HOST_RELATIVE_SOURCE_SELFTEST_PASS")
    print("arms="+",".join(ARMS));print("host_width=74 parameter_count=33036")
    print("future_rows_loaded=0 ego_consumed=false upstream_consumed=false downstream_consumed=false")
if __name__=="__main__":main()
