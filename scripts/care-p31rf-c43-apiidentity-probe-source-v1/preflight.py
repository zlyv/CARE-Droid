#!/usr/bin/env python3
from pathlib import Path
import csv, gzip, json
import numpy as np

ROOT=Path("artifacts/care-p31rf-c43-apiidentity-probe-source-v1")
REF=Path("artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE/source_predictions.csv")
PRED=Path("artifacts/care-p31-stage1-rf-selection-source-v1/arms/C_RF_RFROUTE/source_predictions.csv")
ENDPOINT=Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz")
CACHE=Path("artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz")

def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))

def main():
    required=(REF,PRED,ENDPOINT,CACHE)
    missing=[str(p) for p in required if not p.exists() or p.stat().st_size==0]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
    ref=rows(REF); pred=rows(PRED)
    if len(ref)!=799 or len(pred)!=799:raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort is not 799")
    if {r["sha256"].casefold() for r in ref}!={r["sha256"].casefold() for r in pred}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source prediction cohorts differ")
    if sum(int(float(r["A25__routed"])) for r in ref)!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: route is not 200")
    if {int(float(r["fold"])) for r in ref}!={1,2,3,4,5}:raise RuntimeError("BLOCKED_BY_CONTRACT: fold contract changed")
    endpoint_n=0
    with gzip.open(ENDPOINT,"rt",encoding="utf-8") as h:
        for line in h:
            if line.strip() and json.loads(line).get("status")=="OK":endpoint_n+=1
    if endpoint_n<799:raise RuntimeError("BLOCKED_BY_CONTRACT: endpoint cache has fewer than 799 OK rows")
    with np.load(CACHE,allow_pickle=False) as z:
        if "sha256" not in z or len(z["sha256"])!=799:raise RuntimeError("BLOCKED_BY_CONTRACT: C43 cache cohort changed")
        if not any(v.ndim==2 and v.shape[1]==105 for v in (z[k] for k in z.files)):
            raise RuntimeError("BLOCKED_BY_CONTRACT: no Compact43+Role31+Relation31 matrix")
    ROOT.mkdir(parents=True,exist_ok=True)
    protocol={"source_only":True,"year":2017,"source_rows":799,"future_rows_loaded":0,
      "stage1":"P31 calibrated RF only","router":"RF-anchored LR/RF/XGB/SVM MeanOpposition Top25%",
      "route_budget":0.25,"route_rows":200,"fixed_folds":5,"threshold":0.5,"max_delta":2.25,
      "neural_training_invoked":False,"api_identity":"stable hash of canonical sensitive API signature",
      "api_names_readable":False,"vocabulary":"outer-train only","min_apk_df":2,"max_vocab":2048,
      "api_projection":"outer-train TruncatedSVD padded to 31","validation_unseen":"UNK plus retained Role fallback",
      "pair_or_path_extracted":False,"advance_order":"E1 must pass before E2/E3"}
    (ROOT/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8")
    print("P31RF_C43_APIIDENTITY_PREFLIGHT_PASS")
    print("source_rows=799 route=200 future_rows_loaded=0 neural_training_invoked=false")
    print("api_identity=stable_signature_hash raw_api_names_persisted=false")

if __name__=="__main__":main()

