#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from role_api_core import *
ROOT=Path("artifacts/care-role-conditioned-api-v1"); BASE=Path("artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz"); ENDPOINT=Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"); REF=Path("artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE/source_predictions.csv")
def read_rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def main():
    with np.load(BASE,allow_pickle=False) as z:data={k:np.asarray(z[k]) for k in z.files}
    sha=np.asarray([norm(x) for x in data["sha256"]]); values=np.asarray(data["host_values"],np.float32); off=np.asarray(data["host_offsets"],int); real=np.asarray(data["real_host_count"],int)
    if len(sha)!=799 or values.shape[1]!=105 or off.shape!=(800,):raise RuntimeError("BLOCKED_BY_CONTRACT: frozen C43 cache changed")
    endpoint=read_endpoint(ENDPOINT); ref_by={norm(r["sha256"]):r for r in read_rows(REF)}; api=[];roles=[];api_off=[0];host_apk=[];audit=[];role_stats=[{"role":i,"role_name":ROLE_NAMES[i],"api_occurrences":0,"unique":set(),"apk":set(),"benign_apk":set(),"malware_apk":set(),"routed_error_apk":set()} for i in range(ROLE_N)]; row_id=0
    appended=[]
    for ai,key in enumerate(sha):
        hosts=endpoint_hosts(endpoint[key]); n=int(real[ai]); stored=int(off[ai+1]-off[ai])
        if len(hosts)!=n or stored!=max(n,1):raise RuntimeError(f"BLOCKED_BY_CONTRACT: Host order/count mismatch {key} endpoint={len(hosts)} frozen={n}")
        ids=[];occ=0;uniq=set();per=np.zeros(ROLE_N,int)
        if n:
            for host in hosts:
                ids.append(row_id);host_apk.append(ai);row_id+=1
                for token,role in host["pairs"]:
                    api.append(token);roles.append(role);occ+=1;uniq.add(token);per[role]+=1;role_stats[role]["api_occurrences"]+=1;role_stats[role]["unique"].add(token);role_stats[role]["apk"].add(ai)
                    y=int(float(ref_by[key]["label"]));role_stats[role]["benign_apk" if y==0 else "malware_apk"].add(ai)
                    base=float(ref_by[key]["Global24__prob"]);route=int(float(ref_by[key]["A25__routed"]))==1
                    if route and ((base>=.5)!=y):role_stats[role]["routed_error_apk"].add(ai)
                api_off.append(len(api))
        else:
            # Preserve the frozen zero-Host placeholder as an auditable row so
            # every validation APK has a feature-norm record, while its API
            # registry entry remains exactly empty.
            ids=[row_id];host_apk.append(ai);row_id+=1;api_off.append(len(api))
        payload=endpoint[key];node_base=np.asarray(payload.get("node_base_features") or [],np.float32);graph_nodes=len(payload.get("node_ids") or []);graph_edges=len((payload.get("edge_index") or [[],[]])[0]);method_count=int(np.sum(node_base[:,4]<=.5)) if node_base.ndim==2 and len(node_base) else 0
        block=np.asarray(ids,np.float32)[:,None];appended.append(block);audit.append({"apk_index":ai,"sha256":key,"real_host_count":n,"sensitive_api_occurrence_count":occ,"unique_api_count":len(uniq),"role_count":int(np.sum(per>0)),"method_count":method_count,"graph_node_count":graph_nodes,"graph_edge_count":graph_edges,**{f"role_api_count_{r}":int(per[r]) for r in range(ROLE_N)}})
    side=np.concatenate(appended); raw=np.concatenate([values,side],1); out=ROOT/"cache";out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"role_api_raw_source.npz",sha256=data["sha256"],host_values=raw,host_offsets=off,real_host_count=real,host_was_empty=data["host_was_empty"])
    np.savez_compressed(out/"role_api_registry.npz",api_offsets=np.asarray(api_off,np.int64),api_ids=np.asarray(api,dtype="U32"),api_roles=np.asarray(roles,np.int8),host_apk_index=np.asarray(host_apk,np.int32))
    write_csv(out/"apk_api_audit.csv",audit);stat=[]
    for r in role_stats:stat.append({k:(len(v) if isinstance(v,set) else v) for k,v in r.items()})
    write_csv(out/"role_api_statistics.csv",stat)
    summary={"source_rows":799,"real_hosts":len(host_apk),"api_role_occurrences":len(api),"unique_api_ids":len(set(api)),"stored_width":106,"host_order_contract":"sorted endpoint host index equals frozen C43 Host order","future_rows_loaded":0}
    (out/"cache_audit.json").write_text(json.dumps(summary,indent=2)+"\n")
    for n,label in ((5,"FIVE"),(25,"TWENTYFIVE")):
        sample=audit[:n];print(f"ROLE_API_{label}_APK_AUDIT PASS rows={n} hosts={sum(int(r['real_host_count']) for r in sample)} api_occurrences={sum(int(r['sensitive_api_occurrence_count']) for r in sample)}")
    print("ROLE_API_CACHE_PASS");print(json.dumps(summary))
if __name__=="__main__":main()
