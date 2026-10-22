#!/usr/bin/env python3
"""Build three source-only Host-V2 caches from frozen label-free assets."""
from __future__ import annotations

import argparse, csv, gzip, hashlib, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from host_relative_features import ARMS, build_arm_values, reconstruct_hosts

SOURCE = Path("artifacts/care-source-local-assets-v2/regions_2017.npz")
ENDPOINT = Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz")
ROOT = Path("artifacts/care-di25-host-relative-source-v1")
MANIFESTS = (
    Path("artifacts/care-source-local-assets-v2/local_asset_manifest.json"),
    Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json"),
)

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def locate_manifest():
    for path in MANIFESTS:
        if path.is_file():return path
    raise RuntimeError("BLOCKED_BY_CONTRACT: local_asset_manifest.json not found")

def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def rho(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if np.std(x)<1e-12 or np.std(y)<1e-12:return float("nan")
    return float(spearmanr(x,y).statistic)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=ROOT);args=parser.parse_args()
    manifest_path=locate_manifest();manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    mapping=np.asarray(manifest.get("payload_to_model_role"),dtype=np.int64)
    role_order=manifest.get("model_role_order")
    if mapping.shape!=(31,) or sorted(mapping.tolist())!=list(range(31)) or not isinstance(role_order,list) or len(role_order)!=31:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen role mapping contract changed")
    with np.load(SOURCE,allow_pickle=False) as z:
        ids=[str(x).lower() for x in z["sha256"]];old=np.asarray(z["host_values"],np.float32);offsets=np.asarray(z["host_offsets"],np.int64);real=np.asarray(z["real_host_count"],np.int64);empty=np.asarray(z["host_was_empty"],np.int8)
    if len(ids)!=799 or len(set(ids))!=799 or old.shape!=(95316,74) or offsets.shape!=(800,) or int(real.sum())!=95314:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source Host asset changed")
    payloads={};duplicates=[];rows=0
    with gzip.open(ENDPOINT,"rt",encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            rows+=1;row=json.loads(line);sid=str(row.get("sha256") or row.get("sample_id") or "").lower()
            if sid in payloads:duplicates.append(sid)
            if sid and row.get("payload") is not None:payloads[sid]=row["payload"]
    if rows!=799 or duplicates or set(payloads)!=set(ids):
        raise RuntimeError("BLOCKED_BY_CONTRACT: endpoint cache lineage changed")
    arm_rows={arm:[] for arm in ARMS};per_apk=[]
    for index,sid in enumerate(ids):
        payload=payloads[sid];records=reconstruct_hosts(payload,mapping);begin,end=int(offsets[index]),int(offsets[index+1]);real_n=int(real[index])
        if len(records)!=real_n:raise RuntimeError(f"BLOCKED_BY_CONTRACT: real Host mismatch {sid}")
        if real_n:
            source_real=old[begin:begin+real_n]
            for arm in ARMS:arm_rows[arm].append(build_arm_values(source_real,records,arm))
        else:
            if end-begin!=1 or int(empty[index])!=1:raise RuntimeError(f"BLOCKED_BY_CONTRACT: placeholder mismatch {sid}")
            for arm in ARMS:arm_rows[arm].append(old[begin:end].copy())
        # The means of every transformed block must be zero within non-empty APKs.
        if real_n:
            hm=arm_rows["HM_R"][-1][:,31:62];hc=arm_rows["HC_R"][-1][:,62:74]
            if float(np.max(np.abs(hm.mean(0))))>2e-6 or float(np.max(np.abs(hc.mean(0))))>2e-6:
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: within-APK centering failed {sid}")
        item={"sha256":sid,"real_hosts":real_n,"placeholder":int(real_n==0),"graph_nodes":len(payload.get("node_ids") or []),"graph_edges":len((payload.get("edge_index") or [[],[]])[0])}
        for arm in ARMS:
            bag=arm_rows[arm][-1][:real_n] if real_n else np.empty((0,74),np.float32)
            for block,left,right in (("M",31,62),("C",62,74)):
                values=bag[:,left:right]
                item[f"{arm}_{block}_signed_mean"]=float(values.mean()) if values.size else 0.0
                item[f"{arm}_{block}_mean_abs"]=float(np.abs(values).mean()) if values.size else 0.0
                item[f"{arm}_{block}_max_abs"]=float(np.abs(values).max()) if values.size else 0.0
        per_apk.append(item)
    args.output.mkdir(parents=True,exist_ok=True);cache_meta={}
    for arm in ARMS:
        values=np.concatenate(arm_rows[arm],axis=0).astype(np.float32)
        if values.shape!=(95316,74):raise RuntimeError(f"BLOCKED_BY_CONTRACT: {arm} shape={values.shape}")
        path=args.output/"caches"/(arm+"_regions_2017.npz");path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,sha256=np.asarray(ids),host_values=values,host_offsets=offsets,real_host_count=real,host_was_empty=empty)
        cache_meta[arm]={"path":str(path),"sha256":sha256(path),"shape":list(values.shape)}
    size_rows=[]
    nonempty=[r for r in per_apk if not r["placeholder"]]
    for feature in [k for k in per_apk[0] if k.endswith(("signed_mean","mean_abs","max_abs"))]:
        x=[r[feature] for r in nonempty]
        size_rows.append({"feature":feature,"rho_graph_nodes":rho(x,[r["graph_nodes"] for r in nonempty]),"rho_graph_edges":rho(x,[r["graph_edges"] for r in nonempty]),"rho_real_hosts":rho(x,[r["real_hosts"] for r in nonempty])})
    write_csv(args.output/"transformed_per_apk_audit.csv",per_apk);write_csv(args.output/"transformed_size_association.csv",size_rows)
    transformed_signed=("HM_R_M_signed_mean","HC_R_C_signed_mean","HMC_R_M_signed_mean","HMC_R_C_signed_mean")
    max_signed=max(abs(r[k]) for r in per_apk for k in transformed_signed)
    if max_signed>2e-6:raise RuntimeError("BLOCKED_BY_CONTRACT: transformed signed means are not centered")
    audit={"source_rows":799,"year":2017,"real_hosts":95314,"placeholder_rows":2,"host_width":74,"payload_to_model_role_applied":True,"manifest":str(manifest_path),"manifest_sha256":sha256(manifest_path),"endpoint_cache_sha256":sha256(ENDPOINT),"source_region_sha256":sha256(SOURCE),"labels_loaded":False,"future_rows_loaded":0,"ego_consumed":False,"upstream_consumed":False,"downstream_consumed":False,"within_apk_signed_mean_max_abs":max_signed,"caches":cache_meta}
    (args.output/"cache_build_audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print("HOST_RELATIVE_CACHES_PASS")
    print("source_rows=799 real_hosts=95314 placeholder_rows=2 host_width=74")
    print("payload_to_model_role_applied=true")
    print("future_rows_loaded=0 ego_consumed=false upstream_consumed=false downstream_consumed=false")
if __name__=="__main__":main()
