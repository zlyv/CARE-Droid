#!/usr/bin/env python3
from audit_core import reconstruct
p={"node_ids":["h0","h1","e0","e1"],"node_role_ids":[[],[],[2],[2,3]],"node_base_features":[[0]*15 for _ in range(4)],"edge_index":[[0,0,1],[2,3,0]],"edge_features":[[0,1,0,1,0,1,0],[1,0,1,1,0,0,1],[0,0,0,0,0,0,0]]}
p["node_base_features"][2][4]=1;p["node_base_features"][3][4]=1
r,n,e=reconstruct(p)
assert n==4 and e==3 and len(r)==1
h=r[0];assert h["unique_endpoint_count"]==2 and h["role_endpoint_counts"][2]==2 and h["role_endpoint_counts"][3]==1
assert h["reflection_ratio"]==.5 and h["icc_ratio"]==.5 and h["non_direct_ratio"]==.5
print("HOST_INFORMATION_LOSS_SELFTEST_PASS")
print("training_invoked=false")
print("apk_reanalysis_invoked=false")
print("labels_loaded=false")
print("future_rows_loaded=0")
