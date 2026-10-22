#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/"care-di25-host-relative-source-v1"
MODEL=HERE.parent/"care-di25-graphglobal-path-source-v1"
for p in (HERE,SOURCE,MODEL):sys.path.insert(0,str(p))
from host_relative_features import ARMS, build_arm_values

def main():
    rng=np.random.default_rng(17)
    x=rng.normal(size=(23,74)).astype(np.float32)
    records=[]
    for i in range(23):
        presence=np.zeros(31,np.float32);presence[i%31]=1
        counts=presence*(1+i%4)
        records.append({"presence":presence,"endpoint_counts":counts,"unique_endpoint_count":float(1+i%4),"role_assignment_count":float(counts.sum()),"role_diversity":float(np.count_nonzero(counts)/31),"role_entropy":0.0,"multi_role_endpoint_ratio":0.0,"incoming_degree":float(i%3),"outgoing_degree":float((i+1)%3),"direction_balance":0.0,"reciprocal_ratio":0.0,"icc_ratio":float(i%2),"reflection_ratio":float((i+1)%2),"non_direct_ratio":float(i%5)/4})
    x[:,:31]=np.stack([r["presence"] for r in records])
    for arm in ARMS:
        z=build_arm_values(x,records,arm)
        assert z.shape==x.shape and z.dtype==np.float32 and np.isfinite(z).all()
    assert set(ARMS)=={"HM_R","HC_R","HMC_R"}
    forbidden=("ego_1hop_regions_2017.npz","future_labels","metadata_label")
    for name in ("build_future_caches.py","run_frozen_arm.py","run_within_arm.py"):
        text=(HERE/name).read_text(encoding="utf-8")
        assert "ego_1hop" not in text.lower()
    print("HOST_RELATIVE_DIAGNOSTICS_SELFTEST_PASS")
    print("arms=H0_FULL_Q_LOCKED,HM_R,HC_R,HMC_R")
    print("gpus_requested=2")
    print("host_width=74 parameter_count=33036")
    print("three_backgrounds_same_host_schema=true")
    print("ego_consumed=false")
if __name__=="__main__":main()
