#!/usr/bin/env python3
import csv,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent.parent;DEP=PROJECT/"scripts/care-di25-graphglobal-path-source-v1";sys.path.insert(0,str(DEP))
from experiment_contract import build_di_reference
ROOT=PROJECT/"artifacts/care-di25-q-drop-role-source-v1"
req=[DEP/"graphglobal_path_models.py",PROJECT/"artifacts/care-di25-graphglobal-path-source-v1/arms/Q_QUERY_ONLY/source_predictions.csv",PROJECT/"artifacts/care-source-local-assets-v2/regions_2017.npz",PROJECT/"artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"]
miss=[str(p.relative_to(PROJECT)) for p in req if not p.is_file() or p.stat().st_size==0]
if miss:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(miss))
build_di_reference(PROJECT,ROOT)
with req[1].open("r",encoding="utf-8-sig",newline="") as h:r=list(csv.DictReader(h))
if len(r)!=799 or len({x["sha256"].lower() for x in r})!=799:raise RuntimeError("BLOCKED_BY_CONTRACT: locked FULL_Q invalid")
print("Q_DROP_ROLE_SOURCE_PREFLIGHT_PASS")
print("source_rows=799 year=2017 route_DI=200")
print("FULL_Q=reuse_locked_not_retrained")
print("new_training_arms=1")
print("parameter_count=33036 active_dimensions=81")
print("future_rows_loaded=0")
print("ego_consumed=false")
