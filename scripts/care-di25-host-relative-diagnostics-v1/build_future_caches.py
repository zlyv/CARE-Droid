#!/usr/bin/env python3
"""Build label-free 2018-2021 Host-relative caches from frozen assets."""
from __future__ import annotations
import gzip,hashlib,json,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent.parent;SOURCE_CODE=PROJECT/"scripts/care-di25-host-relative-source-v1"
sys.path.insert(0,str(SOURCE_CODE))
from host_relative_features import ARMS,build_arm_values,reconstruct_hosts

ROOT=Path("artifacts/care-di25-host-relative-diagnostics-v1")
REGION_CANDIDATES=(Path("artifacts/care-future-local-assets-v2/regions_2018_2021.npz"),Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/regions_2018_2021.npz"))
ENDPOINT_CANDIDATES=(Path("artifacts/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"),Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"))
MANIFEST_CANDIDATES=(Path("artifacts/care-source-local-assets-v2/local_asset_manifest.json"),Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json"))
def locate(candidates,name):
    for p in candidates:
        if p.is_file() and p.stat().st_size>100:return p
    raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty "+name)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def main():
    region=locate(REGION_CANDIDATES,"future regions");endpoint=locate(ENDPOINT_CANDIDATES,"future endpoint cache");manifest=locate(MANIFEST_CANDIDATES,"role manifest")
    meta=json.loads(manifest.read_text(encoding="utf-8"));mapping=np.asarray(meta.get("payload_to_model_role"),np.int64)
    if mapping.shape!=(31,) or sorted(mapping.tolist())!=list(range(31)):raise RuntimeError("BLOCKED_BY_CONTRACT: role mapping changed")
    with np.load(region,allow_pickle=False) as z:
        ids=[str(x).lower() for x in z["sha256"]];old=np.asarray(z["host_values"],np.float32);offsets=np.asarray(z["host_offsets"],np.int64);real=np.asarray(z["real_host_count"],np.int64);empty=np.asarray(z["host_was_empty"],np.int8)
    if len(ids)!=1989 or len(set(ids))!=1989 or offsets.shape!=(1990,) or old.ndim!=2 or old.shape[1]!=74 or int(real.sum())!=263600 or len(old)!=263602:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: future region contract changed ids={len(ids)} shape={old.shape} real={int(real.sum())}")
    payloads={};rows=0;duplicates=[]
    with gzip.open(endpoint,"rt",encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            rows+=1;r=json.loads(line);sid=str(r.get("sha256") or r.get("sample_id") or "").lower()
            if sid in payloads:duplicates.append(sid)
            if sid and r.get("payload") is not None:payloads[sid]=r["payload"]
    if rows!=1989 or duplicates or set(payloads)!=set(ids):raise RuntimeError("BLOCKED_BY_CONTRACT: future endpoint lineage changed")
    output={a:[] for a in ARMS}
    for i,sid in enumerate(ids):
        rec=reconstruct_hosts(payloads[sid],mapping);b,e=int(offsets[i]),int(offsets[i+1]);n=int(real[i])
        if len(rec)!=n:raise RuntimeError(f"BLOCKED_BY_CONTRACT: Host mismatch {sid}")
        if n:
            source=old[b:b+n]
            for arm in ARMS:output[arm].append(build_arm_values(source,rec,arm))
        else:
            if e-b!=1 or int(empty[i])!=1:raise RuntimeError(f"BLOCKED_BY_CONTRACT: placeholder mismatch {sid}")
            for arm in ARMS:output[arm].append(old[b:e].copy())
        if (i+1)%200==0 or i+1==len(ids):print(f"future Host-relative reconstruction {i+1}/{len(ids)}",flush=True)
    cache={}
    for arm in ARMS:
        values=np.concatenate(output[arm]).astype(np.float32)
        if values.shape!=(263602,74):raise RuntimeError(f"BLOCKED_BY_CONTRACT: {arm} shape {values.shape}")
        p=ROOT/"caches"/(arm+"_regions_2018_2021.npz");p.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(p,sha256=np.asarray(ids),host_values=values,host_offsets=offsets,real_host_count=real,host_was_empty=empty)
        cache[arm]={"path":str(p),"sha256":digest(p),"shape":list(values.shape)}
    audit={"future_rows":1989,"years":[2018,2019,2020,2021],"real_hosts":263600,"placeholder_rows":2,"host_width":74,"region_source":str(region),"region_sha256":digest(region),"endpoint_source":str(endpoint),"endpoint_sha256":digest(endpoint),"manifest":str(manifest),"manifest_sha256":digest(manifest),"payload_to_model_role_applied":True,"labels_loaded":False,"ego_consumed":False,"upstream_consumed":False,"downstream_consumed":False,"caches":cache}
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/"future_cache_build_audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print("HOST_RELATIVE_FUTURE_CACHES_PASS");print("future_rows=1989 real_hosts=263600 placeholder_rows=2 host_width=74");print("labels_loaded=false ego_consumed=false")
if __name__=="__main__":main()
