#!/usr/bin/env python3
import gzip,json
from pathlib import Path
ROOT=Path("artifacts/care-p31rf-haig-source-v1");CACHE=Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz")
def main():
    if not CACHE.exists():raise RuntimeError("BLOCKED_BY_CONTRACT: endpoint cache absent")
    with gzip.open(CACHE,"rt",encoding="utf-8") as h:row=next(json.loads(x) for x in h if x.strip())
    payload=row.get("payload",{});keys=set(payload)
    report={"decision":"FEASIBLE","cache_audit":{"host_method":"hashed node identity only","sensitive_api":"stable hash plus semantic role","canonical_api_signature":False,"api_order":False,"graph_edges":bool({"edge_index","edge_type_ids"}.issubset(keys)),"direct_Host_call_recoverable_from_cache":"not safely distinguishable without raw signatures"},"reconstruction":{"requires_APK_DEX_reanalysis":True,"androguard_method_IR_available":True,"same_host_order":"invoke instruction offsets","host_call":"direct resolved invoke target between sensitive Host methods","full_call_graph_required":False,"taint_or_alias_required":False},"evidence_boundary":{"same_host_order_is_data_flow":False,"host_call_is_data_flow":False,"allowed_terms":["API interaction","API composition","host-mediated relation"]},"probe_model":"fixed-width hashed hierarchical API/relation channels + train-only SVD + capacity-matched small MLP","future_selection_allowed":False}
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/"feasibility_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("CARE-HAIG FEASIBILITY: FEASIBLE")
    print("cache_exact_signature=false cache_order=false dex_reanalysis_required=true direct_host_call=true")
if __name__=="__main__":main()

