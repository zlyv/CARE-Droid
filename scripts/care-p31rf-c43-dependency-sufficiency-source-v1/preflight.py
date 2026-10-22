#!/usr/bin/env python3
import csv, json, sys
from pathlib import Path

ROOT=Path("artifacts/care-p31rf-c43-dependency-sufficiency-source-v1")
REF=Path("artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE/source_predictions.csv")
PRED=Path("artifacts/care-p31-stage1-rf-selection-source-v1/arms/C_RF_RFROUTE/source_predictions.csv")
ENDPOINT=Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz")
P71=Path("artifacts/p71/verified_dependency_features_2017_v3.csv")
API=Path("scripts/care-p31rf-c43-apiidentity-probe-source-v1/api_identity_core.py")

def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))

def main():
    required=(REF,PRED,ENDPOINT,P71,API)
    missing=[str(p) for p in required if not p.exists() or p.stat().st_size==0]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
    ref=rows(REF);dep=rows(P71)
    if len(ref)!=799:raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort is not 799")
    ids={r["sha256"].casefold() for r in ref}; dep_ids={r["sha256"].casefold() for r in dep}
    if not ids.issubset(dep_ids):raise RuntimeError(f"BLOCKED_BY_CONTRACT: P71 lacks {len(ids-dep_ids)} source rows")
    if sum(int(float(r["A25__routed"])) for r in ref)!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: RF route is not 200")
    if {int(float(r["fold"])) for r in ref}!={1,2,3,4,5}:raise RuntimeError("BLOCKED_BY_CONTRACT: fold contract changed")
    ROOT.mkdir(parents=True,exist_ok=True)
    protocol={
      "source_only":True,"year":2017,"source_rows":799,"future_rows_loaded":0,
      "stage1":"P31 calibrated RF only","router":"RF-anchored LR/RF/XGB/SVM MeanOpposition Top25%",
      "route_rows":200,"threshold":0.5,"fixed_folds":5,"max_delta_unchanged":2.25,
      "neural_training_invoked":False,"c43_modified":False,
      "candidate":"verified result-centred role dependency sufficiency probe",
      "dependency_evidence":["local register def-use","interprocedural argument/return","resolved field write/read"],
      "not_dependency":["same method","same class","call reachability","co-occurrence"],
      "representations":["D0_ROLE","D1_ROLE_API","D2_ROLE_COOCCURRENCE","D3_ROLE_VERIFIED_DEP","D4_COUNT_ONLY","D5_ROLE_SHUFFLED_DEP"],
      "important_limitation":"P71 preserves source/sink semantic roles, not exact source API identity or C43 Host ownership; a pass authorizes Host-aligned re-extraction, not C43 integration.",
      "advance_gate":{"min_FP_direction_gain":0.10,"max_FN_direction_drop":0.05,"overall_gain":True,"fp_vs_tp_auc_min":0.55,"beats_cooccurrence":True,"beats_shuffled":True,"nonnegative_folds":3}
    }
    (ROOT/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8")
    print("P31RF_DEPENDENCY_SUFFICIENCY_PREFLIGHT_PASS")
    print("source_rows=799 route=200 future_rows_loaded=0 c43_modified=false")
    print("dependency_level=verified_role_pair exact_source_api_and_host_owner_not_persisted")

if __name__=="__main__":main()

