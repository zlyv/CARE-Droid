#!/usr/bin/env python3
from pathlib import Path
import json,numpy as np
from host_sidecar import norm,read_endpoint,load_graph,align_one
BASE=Path("artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz")
ENDPOINT=Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz")
GRAPHS=Path("artifacts/care-p31rf-haig-source-v1/graphs/source")
OUT=Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1/alignment_probe.json")
def main():
    with np.load(BASE,allow_pickle=False) as z:sha=[norm(x) for x in z["sha256"]]
    endpoint=read_endpoint(ENDPOINT);total=matched=amb=fpeq=0;apk_full=0
    for key in sha:
        _,_,_,a=align_one(endpoint[key],load_graph(GRAPHS/(key+".json.gz")));total+=a["endpoint_hosts"];matched+=a["matched"];amb+=a["ambiguous_or_unmatched"];fpeq+=a["matched_api_fingerprint_equal"];apk_full+=int(a["matched"]==a["endpoint_hosts"])
    result={"source_rows":len(sha),"endpoint_hosts":total,"direct_id_matched_hosts":matched,"ambiguous_or_unmatched_hosts":amb,"direct_id_coverage":matched/max(total,1),"fully_aligned_apks":apk_full,"api_fingerprint_agreement_among_id_matches":fpeq/max(matched,1),"identity_contract":"endpoint uint64 == int(HAIG host_id first16hex,16)","positional_matching_used":False,"future_rows_loaded":0,"decision":"PASS" if matched/max(total,1)>=.90 else "BLOCKED"}
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+"\n");print("P31RF_HAIG_DIRECT_HOST_ALIGNMENT_"+result["decision"]);print(json.dumps(result,indent=2))
    if result["decision"]!="PASS":raise RuntimeError("BLOCKED_BY_CONTRACT: direct Host identity alignment below 90%; provide exact endpoint extractor revision before training")
if __name__=="__main__":main()
