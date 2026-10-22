#!/usr/bin/env python3
from __future__ import annotations
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path("artifacts/care-litelocal-p31lr-c43-prov6-source-v1")
BASE=Path("artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz")
ENDPOINT=Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz")

def prov(payload):
    base=np.asarray(payload.get("node_base_features") or [],np.float32)
    if base.size==0:base=np.empty((0,15),np.float32)
    roles=payload.get("node_role_ids") or []; edges=payload.get("edge_index") or [[],[]]
    endpoints=set(map(int,np.flatnonzero(base[:,4]>.5))); callers={}
    for s,t in zip(edges[0],edges[1]):
        s=int(s);t=int(t)
        if t in endpoints:callers.setdefault(s,set()).add(t)
    hosts=[]; values=[]
    for h,targets in callers.items():
        # Match the frozen 2017 source loader exactly.  Payloads can contain
        # sentinel/out-of-contract role ids; they must not create a Host.
        merged={int(r) for t in targets for r in (roles[t] or [])
                if 0 <= int(r) < 31}
        if merged:hosts.append(int(h))
    for h in sorted(hosts):
        targets=sorted(callers[h]); selfp=base[h,5:8]; endpoint=np.mean(base[targets,5:8],axis=0)
        values.append(np.concatenate([selfp,endpoint]))
    return np.asarray(values,np.float32).reshape(-1,6)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--base",type=Path,default=BASE);parser.add_argument("--endpoint",type=Path,default=ENDPOINT);a=parser.parse_args()
    with np.load(a.base,allow_pickle=False) as z:data={k:np.asarray(z[k]) for k in z.files}
    sha=np.asarray([str(v).strip().casefold() for v in data["sha256"]]); values=np.asarray(data["host_values"],np.float32); offsets=np.asarray(data["host_offsets"],np.int64); real=np.asarray(data["real_host_count"],np.int64)
    if len(sha)!=799 or values.shape[1]!=105 or offsets.shape!=(800,):raise RuntimeError("BLOCKED_BY_CONTRACT: base C43 cache changed")
    payload={}
    with gzip.open(a.endpoint,"rt",encoding="utf-8") as h:
        for line in h:
            row=json.loads(line); key=str(row.get("sha256") or row.get("sample_id") or "").strip().casefold()
            graph=row.get("payload")
            if not isinstance(graph,dict):
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: endpoint wrapper has no graph payload {key}")
            required={"node_base_features","node_role_ids","edge_index"}
            if not required.issubset(graph):
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: endpoint graph schema mismatch {key}")
            payload[key]=graph
    if set(payload)!=set(sha):raise RuntimeError("BLOCKED_BY_CONTRACT: endpoint/cache cohort mismatch")
    p1=[]; changed=0; real_total=0
    for i,key in enumerate(sha):
        x=prov(payload[key]); n=int(real[i]); stored=int(offsets[i+1]-offsets[i])
        if len(x)!=n or stored!=max(n,1):
            raise RuntimeError(
                f"BLOCKED_BY_CONTRACT: Host order/count mismatch {key} "
                f"reconstructed={len(x)} frozen_real={n} frozen_stored={stored}"
            )
        if n==0:x=np.zeros((1,6),np.float32)
        p1.append(x);real_total+=n
    p1=np.concatenate(p1);p0=np.zeros_like(p1);p2=p1.copy()
    # Shuffle within APK and existing primary-role sidecar. This preserves APK/role marginals.
    for i,key in enumerate(sha):
        lo,hi=int(offsets[i]),int(offsets[i+1]);n=int(real[i])
        if n<2:continue
        role=np.argmax(np.maximum(values[lo:hi,43:74],0),axis=1)
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"little"))
        for r in np.unique(role):
            idx=np.flatnonzero(role==r)
            if len(idx)>1:
                perm=idx[rng.permutation(len(idx))]; before=p2[lo+idx].copy();p2[lo+idx]=p1[lo+perm];changed+=int(np.sum(np.any(before!=p2[lo+idx],axis=1)))
    outdir=ROOT/"cache";outdir.mkdir(parents=True,exist_ok=True)
    for arm,extra in (("P0_ZERO6",p0),("P1_PROV6",p1),("P2_SHUFFLED_PROV6",p2)):
        np.savez_compressed(outdir/(arm+".npz"),sha256=data["sha256"],host_values=np.concatenate([values,extra],axis=1),host_offsets=offsets,real_host_count=real,host_was_empty=data["host_was_empty"])
    audit={"source_rows":799,"real_hosts":int(real_total),"stored_rows":int(len(values)),"stored_width":111,"prov6_layout":["host_package_aligned","host_bundled_nonpackage","host_external_framework","endpoint_package_aligned_mean","endpoint_bundled_nonpackage_mean","endpoint_external_framework_mean"],"p1_nonzero_rows":int(np.sum(np.any(p1!=0,axis=1))),"p2_changed_rows":int(changed),"shuffle_scope":"within APK and existing primary role","future_rows_loaded":0,"ego_consumed":False}
    (outdir/"cache_audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print("P31LR_PROV6_CACHES_PASS");print(f"source_rows=799 real_hosts={real_total} stored_width=111 p2_changed_rows={changed}");print("future_rows_loaded=0 ego_consumed=false")
if __name__=="__main__":main()
