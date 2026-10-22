#!/usr/bin/env python3
"""Append label-free 31-role endpoint weights to the frozen H0 Host cache."""
from __future__ import annotations
import gzip,json,sys,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent.parent;SRC=PROJECT/"scripts/care-di25-host-relative-source-v1"
sys.path.insert(0,str(SRC));from host_relative_features import reconstruct_hosts
REGION=Path("artifacts/care-source-local-assets-v2/regions_2017.npz");ENDPOINT=Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz");OUT=Path("artifacts/care-di25-roleconditioned-background-source-v1/cache/regions_2017_role_sidecar.npz")
MANIFESTS=(Path("artifacts/care-source-local-assets-v2/local_asset_manifest.json"),Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json"))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def main():
    manifest=next((p for p in MANIFESTS if p.is_file()),None)
    if manifest is None:raise RuntimeError("BLOCKED_BY_CONTRACT: role manifest missing")
    mapping=np.asarray(json.loads(manifest.read_text())["payload_to_model_role"],np.int64)
    with np.load(REGION,allow_pickle=False) as z:ids=[str(x).lower() for x in z["sha256"]];old=np.asarray(z["host_values"],np.float32);offsets=np.asarray(z["host_offsets"],np.int64);real=np.asarray(z["real_host_count"],np.int64);empty=np.asarray(z["host_was_empty"],np.int8)
    payloads={}
    with gzip.open(ENDPOINT,"rt",encoding="utf-8") as f:
        for line in f:
            r=json.loads(line);payloads[str(r.get("sha256") or r.get("sample_id")).lower()]=r["payload"]
    if len(ids)!=799 or old.shape!=(95316,74) or set(ids)!=set(payloads):raise RuntimeError("BLOCKED_BY_CONTRACT: source cache lineage changed")
    rows=[]
    for i,s in enumerate(ids):
        rec=reconstruct_hosts(payloads[s],mapping);b,e=int(offsets[i]),int(offsets[i+1]);n=int(real[i])
        if len(rec)!=n:raise RuntimeError("BLOCKED_BY_CONTRACT: Host order mismatch "+s)
        if n:
            weight=np.stack([r["endpoint_counts"] for r in rec]).astype(np.float32);rows.append(np.concatenate([old[b:b+n],weight],axis=1))
            if not np.allclose(old[b:b+n,:31],(weight>0).astype(np.float32),atol=1e-6,rtol=0):raise RuntimeError("BLOCKED_BY_CONTRACT: role mapping/order mismatch "+s)
        else:rows.append(np.concatenate([old[b:e],np.zeros((1,31),np.float32)],axis=1))
    values=np.concatenate(rows)
    if values.shape!=(95316,105):raise RuntimeError("BLOCKED_BY_CONTRACT: sidecar shape "+str(values.shape))
    OUT.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT,sha256=np.asarray(ids),host_values=values,host_offsets=offsets,real_host_count=real,host_was_empty=empty)
    audit={"source_rows":799,"real_hosts":95314,"placeholder_rows":2,"raw_host_width":74,"role_sidecar_width":31,"stored_width":105,"labels_loaded":False,"future_rows_loaded":0,"ego_consumed":False,"region_sha256":sha(REGION),"endpoint_sha256":sha(ENDPOINT),"output_sha256":sha(OUT)}
    Path("artifacts/care-di25-roleconditioned-background-source-v1/cache_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print("ROLECONDITIONED_BACKGROUND_CACHE_PASS");print("source_rows=799 real_hosts=95314 stored_width=105 labels_loaded=false future_rows_loaded=0")
if __name__=="__main__":main()
