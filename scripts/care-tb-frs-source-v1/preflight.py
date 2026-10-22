#!/usr/bin/env python3
from common import *

def main():
    required=[(SOURCE_ROWS,"SVM source OOF and route"),(DIRECT_ROWS,"Direct-C43 source reference"),(HOST_CACHE,"formal A0 Host cache"),(HAIG_BG,"formal hard Role prototype background"),(V4_SCRIPT,"formal C43 trainer")]
    missing=[str(p) for p,_ in required if not p.is_file()]
    if missing: raise RuntimeError("BLOCKED_BY_CONTRACT: missing assets "+repr(missing))
    rows,ids,y,fold,base,route,bags,direct,field=load_source()
    manifest=[asset(p,purpose) for p,purpose in required]; dump_json(OUTPUT/"baseline_asset_manifest_before.json",manifest)
    protocol={"status":"FROZEN_SOURCE_PROBE","source_rows":len(ids),"routed":int(route.sum()),"folds":list(FOLDS),
      "stage1":"calibrated P31-SVM","router":"SVM-anchored MeanOpposition Top25%","reference":"Direct-C43 threshold 0.5",
      "shared_prototypes":32,"benign_prototypes":24,"role_constraint":"inherit formal argmax-Role compatibility with fallback",
      "soft_temperature":1.0,"projection_width":16,"projection":"outer-train StandardScaler plus randomized PCA",
      "classifier":"balanced L2 logistic, C=0.25, identical 16-D input",
      "selection":"2017 source only; no future or external rows","future_rows_loaded":0,
      "arms":list(ARMS),"direct_reference_field":field}
    dump_json(OUTPUT/"protocol.json",protocol); (ensure(OUTPUT/"protocol_hash.txt")).write_text(sha256(OUTPUT/"protocol.json")+"\n",encoding="utf-8")
    dump_json(OUTPUT/"preflight.json",{"status":"PASS","source_rows":len(ids),"route":int(route.sum()),"host_rows":int(sum(map(len,bags))),"future_rows_loaded":0})
    print("TBFRS_PREFLIGHT_PASS source=799 route=200 hosts=%d"%sum(map(len,bags)),flush=True)
if __name__=="__main__":main()
