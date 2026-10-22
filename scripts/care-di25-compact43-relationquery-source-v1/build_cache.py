#!/usr/bin/env python3
"""Build Compact43 + role-selection31 + relation-query31 source cache."""
from __future__ import annotations
import gzip, hashlib, json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from hostrel_features import reconstruct_host_relations

REGION=Path("artifacts/care-source-local-assets-v2/regions_2017.npz")
ENDPOINT=Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz")
OUT=Path("artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz")
MANIFESTS=(Path("artifacts/care-source-local-assets-v2/local_asset_manifest.json"),Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json"))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def main():
    manifest=next((p for p in MANIFESTS if p.is_file()),None)
    if manifest is None: raise RuntimeError("BLOCKED_BY_CONTRACT: role manifest missing")
    mapping=np.asarray(json.loads(manifest.read_text())["payload_to_model_role"],np.int64)
    with np.load(REGION,allow_pickle=False) as z:
        ids=[str(x).lower() for x in z["sha256"]]; old=np.asarray(z["host_values"],np.float32); offsets=np.asarray(z["host_offsets"],np.int64); real=np.asarray(z["real_host_count"],np.int64); empty=np.asarray(z["host_was_empty"],np.int8)
    payloads={}
    with gzip.open(ENDPOINT,"rt",encoding="utf-8") as f:
        for line in f:
            row=json.loads(line); payloads[str(row.get("sha256") or row.get("sample_id")).lower()]=row["payload"]
    if len(ids)!=799 or old.shape!=(95316,74) or set(ids)!=set(payloads): raise RuntimeError("BLOCKED_BY_CONTRACT: source lineage changed")
    rows=[]; any_neighbor=0; nonzero_relation=0
    for i,sid in enumerate(ids):
        records=reconstruct_host_relations(payloads[sid],mapping); begin,end,count=int(offsets[i]),int(offsets[i+1]),int(real[i])
        if len(records)!=count: raise RuntimeError("BLOCKED_BY_CONTRACT: Host mismatch "+sid)
        if count:
            raw=old[begin:begin+count]
            presence=np.stack([r["presence"] for r in records]).astype(np.float32)
            if not np.allclose(raw[:,:31],presence,atol=1e-6,rtol=0): raise RuntimeError("BLOCKED_BY_CONTRACT: Host order/role mismatch "+sid)
            compact=np.concatenate([presence,raw[:,62:74]],axis=1)
            role_sidecar=np.stack([r["endpoint_counts"] for r in records]).astype(np.float32)
            relation=np.stack([r["directed_neighbor_role"] for r in records]).astype(np.float32)
            rows.append(np.concatenate([compact,role_sidecar,relation],axis=1))
            any_neighbor+=sum((r["incoming_neighbor_count"]+r["outgoing_neighbor_count"])>0 for r in records)
            nonzero_relation+=sum(np.any(r["directed_neighbor_role"]!=0) for r in records)
        else:
            if end-begin!=1 or int(empty[i])!=1: raise RuntimeError("BLOCKED_BY_CONTRACT: placeholder changed")
            compact=np.concatenate([old[begin:end,:31],old[begin:end,62:74]],axis=1)
            rows.append(np.concatenate([compact,np.zeros((1,62),np.float32)],axis=1))
    values=np.concatenate(rows).astype(np.float32)
    if values.shape!=(95316,105): raise RuntimeError("BLOCKED_BY_CONTRACT: cache shape "+str(values.shape))
    OUT.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(OUT,sha256=np.asarray(ids),host_values=values,host_offsets=offsets,real_host_count=real,host_was_empty=empty)
    audit={"source_rows":799,"real_hosts":95314,"placeholder_rows":2,"stored_width":105,"compact_host_width":43,"semantic_selection_sidecar_width":31,"relation_query_width":31,"compact_definition":"SelfRole31+FrozenTail12","duplicate_role_columns_stored":0,"hosts_with_any_host_neighbor":int(any_neighbor),"hosts_with_nonzero_relation":int(nonzero_relation),"labels_loaded":False,"future_rows_loaded":0,"ego_consumed":False,"output_sha256":digest(OUT)}
    Path("artifacts/care-di25-compact43-relationquery-source-v1/cache_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print("COMPACT43_RELATIONQUERY_CACHE_PASS")
    print(f"source_rows=799 real_hosts=95314 relation_nontrivial_hosts={any_neighbor} relation_nonzero_hosts={nonzero_relation}")
    print("compact_host_width=43 duplicate_role_columns_stored=0 future_rows_loaded=0 ego_consumed=false")


if __name__=="__main__": main()
