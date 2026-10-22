#!/usr/bin/env python3
from __future__ import annotations
import gzip,json
from pathlib import Path
import numpy as np
from host_sidecar import *
ROOT=Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1")
BASE=Path("artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz")
ENDPOINT=Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz")
GRAPHS=Path("artifacts/care-p31rf-haig-source-v1/graphs/source")
ARMS=("A0_ROLE_ONLY","A1_API_NO_ROLE","A2_H3_NO_ROLE","A3_ROLE_PLUS_H3","A4_H3_SHUFFLE_NO_ROLE")
def main():
    with np.load(BASE,allow_pickle=False) as z:data={k:np.asarray(z[k]) for k in z.files}
    sha=np.asarray([norm(x) for x in data["sha256"]]);values=np.asarray(data["host_values"],np.float32);off=np.asarray(data["host_offsets"],int);real=np.asarray(data["real_host_count"],int)
    if len(sha)!=799 or values.shape[1]!=105 or off.shape!=(800,):raise RuntimeError("BLOCKED_BY_CONTRACT: frozen C43 cache changed")
    endpoint=read_endpoint(ENDPOINT)
    missing_endpoint=[key for key in sha if key not in endpoint]
    if missing_endpoint:raise RuntimeError(f"BLOCKED_BY_CONTRACT: {len(missing_endpoint)} source SHA lack an OK endpoint payload; first={missing_endpoint[:5]}")
    api_all=[];call_all=[];shuf_all=[];rows=[];total=matched=amb=0
    for i,key in enumerate(sha):
        g=load_graph(GRAPHS/(key+".json.gz"))
        try:api,call,role,audit=align_one(endpoint[key],g)
        except Exception as exc:raise RuntimeError(f"BLOCKED_BY_CONTRACT: Host alignment failed sha={key}: {exc}") from exc
        n=int(real[i]);stored=int(off[i+1]-off[i])
        if len(api)!=n or stored!=max(n,1):raise RuntimeError(f"BLOCKED_BY_CONTRACT: Host count/order mismatch {key} endpoint={len(api)} frozen={n} stored={stored}")
        if n==0:api=np.zeros((1,DIM),np.float32);call=np.zeros((1,DIM),np.float32);role=np.asarray([-1])
        perm=np.arange(len(call))
        for r in np.unique(role):
            idx=np.flatnonzero(role==r)
            if len(idx)>1:perm[idx]=idx[deterministic_shuffle(len(idx),key+"|"+str(r))]
        api_all.append(api);call_all.append(call);shuf_all.append(call[perm]);total+=audit["endpoint_hosts"];matched+=audit["matched"];amb+=audit["ambiguous_or_unmatched"];rows.append({"sha256":key,**audit,"match_rate":audit["matched"]/max(audit["endpoint_hosts"],1)})
    coverage=matched/max(total,1)
    if coverage<.90:raise RuntimeError(f"BLOCKED_BY_CONTRACT: unique Host alignment coverage {coverage:.6f} < 0.90; no positional guessing allowed")
    api=np.concatenate(api_all);call=np.concatenate(call_all);shuf=np.concatenate(shuf_all);zeros=np.zeros_like(api);out=ROOT/"cache";out.mkdir(parents=True,exist_ok=True)
    def make(role_on,api_x,call_x):
        x=values.copy()
        if not role_on:x[:,:31]=0
        return np.concatenate([x,api_x,call_x],axis=1)
    arms={"A0_ROLE_ONLY":make(True,zeros,zeros),"A1_API_NO_ROLE":make(False,api,zeros),"A2_H3_NO_ROLE":make(False,api,call),"A3_ROLE_PLUS_H3":make(True,api,call),"A4_H3_SHUFFLE_NO_ROLE":make(False,api,shuf)}
    for arm,x in arms.items():np.savez_compressed(out/(arm+".npz"),sha256=data["sha256"],host_values=x,host_offsets=off,real_host_count=real,host_was_empty=data["host_was_empty"])
    import csv
    with (out/"host_alignment.csv").open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    audit={"source_rows":799,"real_hosts":total,"uniquely_aligned_hosts":matched,"ambiguous_or_unmatched_hosts":amb,"alignment_coverage":coverage,"stored_width":169,"model_role_policy":"Role31 zeroed only inside Host content for A1/A2/A4; RoleGraph and Relation31 retained for localization","hash_projection":"fixed label-free signed FeatureHasher, L2 normalized, 32 API + 32 Host-call","future_rows_loaded":0}
    (out/"cache_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print("P31RF_C43_HAIG_HOST_CACHES_PASS");print(f"source_rows=799 real_hosts={total} aligned={matched} coverage={coverage:.6f} stored_width=169");print("future_rows_loaded=0")
if __name__=="__main__":main()

