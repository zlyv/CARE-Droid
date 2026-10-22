#!/usr/bin/env python3
"""Build label-free Compact43 + role sidecar + relation-query future cache."""
from __future__ import annotations
import gzip,hashlib,json,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent.parent;SOURCE=PROJECT/"scripts/care-di25-compact43-relationquery-source-v1"
sys.path.insert(0,str(SOURCE));from hostrel_features import reconstruct_host_relations
ROOT=Path("artifacts/care-di25-compact43-relationquery-diagnostics-v1")
OUT=ROOT/"cache/compact43_relation_future.npz"
REGIONS=(Path("artifacts/care-future-local-assets-v2/regions_2018_2021.npz"),Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/regions_2018_2021.npz"))
ENDPOINTS=(Path("artifacts/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"),Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"))
MANIFESTS=(Path("artifacts/care-source-local-assets-v2/local_asset_manifest.json"),Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json"))


def locate(paths,name):
    for path in paths:
        if path.is_file() and path.stat().st_size>100:return path
    raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty "+name)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def main():
    audit_path=ROOT/"future_cache_audit.json"
    if OUT.is_file() and OUT.stat().st_size>100 and audit_path.is_file():
        try:
            with np.load(OUT,allow_pickle=False) as z:
                valid=z["host_values"].shape==(263602,105) and z["host_offsets"].shape==(1990,) and len(z["sha256"])==1989
            if valid:
                print("COMPACT43_RELATIONQUERY_FUTURE_CACHE_PASS reused=true");print("future_rows=1989 real_hosts=263600 labels_loaded=false ego_consumed=false");return
        except Exception:pass
    region=locate(REGIONS,"future regions");endpoint=locate(ENDPOINTS,"future endpoint cache");manifest=locate(MANIFESTS,"role manifest")
    mapping=np.asarray(json.loads(manifest.read_text())["payload_to_model_role"],np.int64)
    with np.load(region,allow_pickle=False) as z:
        ids=[str(x).lower() for x in z["sha256"]];old=np.asarray(z["host_values"],np.float32);offsets=np.asarray(z["host_offsets"],np.int64);real=np.asarray(z["real_host_count"],np.int64);empty=np.asarray(z["host_was_empty"],np.int8)
    if len(ids)!=1989 or len(set(ids))!=1989 or old.shape!=(263602,74) or int(real.sum())!=263600:raise RuntimeError("BLOCKED_BY_CONTRACT: future region contract changed")
    payloads={}
    with gzip.open(endpoint,"rt",encoding="utf-8") as f:
        for line in f:
            row=json.loads(line);sid=str(row.get("sha256") or row.get("sample_id") or "").lower()
            if sid in payloads:raise RuntimeError("BLOCKED_BY_CONTRACT: duplicate endpoint "+sid)
            payloads[sid]=row["payload"]
    if set(payloads)!=set(ids):raise RuntimeError("BLOCKED_BY_CONTRACT: endpoint lineage changed")
    rows=[];neighbors=0;nonzero=0
    for i,sid in enumerate(ids):
        records=reconstruct_host_relations(payloads[sid],mapping);begin,end,count=int(offsets[i]),int(offsets[i+1]),int(real[i])
        if len(records)!=count:raise RuntimeError("BLOCKED_BY_CONTRACT: Host mismatch "+sid)
        if count:
            raw=old[begin:begin+count];presence=np.stack([r["presence"] for r in records]).astype(np.float32)
            if not np.allclose(raw[:,:31],presence,atol=1e-6,rtol=0):raise RuntimeError("BLOCKED_BY_CONTRACT: Host order/role mismatch "+sid)
            compact=np.concatenate([presence,raw[:,62:74]],axis=1);roles=np.stack([r["endpoint_counts"] for r in records]).astype(np.float32);relation=np.stack([r["directed_neighbor_role"] for r in records]).astype(np.float32)
            rows.append(np.concatenate([compact,roles,relation],axis=1));neighbors+=sum((r["incoming_neighbor_count"]+r["outgoing_neighbor_count"])>0 for r in records);nonzero+=sum(np.any(r["directed_neighbor_role"]!=0) for r in records)
        else:
            if end-begin!=1 or int(empty[i])!=1:raise RuntimeError("BLOCKED_BY_CONTRACT: placeholder changed "+sid)
            compact=np.concatenate([old[begin:end,:31],old[begin:end,62:74]],axis=1);rows.append(np.concatenate([compact,np.zeros((1,62),np.float32)],axis=1))
        if (i+1)%200==0 or i+1==len(ids):print(f"future Compact43 relation reconstruction {i+1}/{len(ids)}",flush=True)
    values=np.concatenate(rows).astype(np.float32)
    if values.shape!=(263602,105):raise RuntimeError("BLOCKED_BY_CONTRACT: cache shape "+str(values.shape))
    OUT.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT,sha256=np.asarray(ids),host_values=values,host_offsets=offsets,real_host_count=real,host_was_empty=empty)
    audit={"future_rows":1989,"real_hosts":263600,"placeholder_rows":2,"stored_width":105,"compact_host_width":43,"duplicate_role_columns":0,"relation_query_width":31,"hosts_with_any_host_neighbor":int(neighbors),"hosts_with_nonzero_relation":int(nonzero),"labels_loaded":False,"ego_consumed":False,"output_sha256":digest(OUT)}
    ROOT.mkdir(parents=True,exist_ok=True);audit_path.write_text(json.dumps(audit,indent=2)+"\n")
    print("COMPACT43_RELATIONQUERY_FUTURE_CACHE_PASS reused=false")
    print(f"future_rows=1989 real_hosts=263600 relation_nontrivial_hosts={neighbors} relation_nonzero_hosts={nonzero}")
    print("labels_loaded=false ego_consumed=false")


if __name__=="__main__":main()
