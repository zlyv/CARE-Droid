#!/usr/bin/env python3
from pathlib import Path
import csv,json
ROOT=Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1")
REQ=(Path("artifacts/care-p31rf-haig-source-v1/graphs/source"),Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"),Path("artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz"),Path("artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE/source_predictions.csv"),Path("scripts/care-p31-stage1-rf-selection-source-v1/nested_stage1.py"),Path("scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py"))
def main():
    missing=[str(p) for p in REQ if not p.exists()]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing inputs:\n- "+"\n- ".join(missing))
    with REQ[3].open(encoding="utf-8-sig",newline="") as h:r=list(csv.DictReader(h))
    if len(r)!=799 or sum(int(x["A25__routed"]) for x in r)!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: P31-RF/DI25 contract changed")
    protocol={"source_rows":799,"year":2017,"folds":[160,160,160,160,159],"stage1":"P31 calibrated RF only","router":"RF-anchored MeanOpposition Top25%","route":200,"role_ablation_scope":"Host content only","RoleGraph_localization_retained":True,"max_delta":2.25,"future_rows_loaded":0,"selection_uses_future":False}
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n");print("P31RF_C43_HAIG_HOST_ABLATION_PREFLIGHT_PASS");print("source_rows=799 route=200 future_rows_loaded=0")
if __name__=="__main__":main()

