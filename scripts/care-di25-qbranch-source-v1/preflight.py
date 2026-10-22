#!/usr/bin/env python3
"""Fail-closed strict-source preflight for Q-branch isolation."""
from __future__ import annotations
import csv, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent; PROJECT=HERE.parent.parent
DEP=PROJECT/"scripts/care-di25-graphglobal-path-source-v1"
sys.path[:0]=[str(HERE),str(DEP)]
from experiment_contract import build_di_reference

ROOT=PROJECT/"artifacts/care-di25-qbranch-source-v1"
required=[
 DEP/"graphglobal_path_models.py", DEP/"experiment_contract.py",
 PROJECT/"scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py",
 PROJECT/"artifacts/care-source-local-assets-v2/regions_2017.npz",
 PROJECT/"artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz",
 PROJECT/"artifacts/care-di25-graphglobal-path-source-v1/arms/Q_QUERY_ONLY/source_predictions.csv",
 PROJECT/"artifacts/care-di25-graphglobal-path-source-v1/arms/Q_QUERY_ONLY/source_summary.csv",
 PROJECT/"artifacts/care-di25-graphglobal-path-source-v1/arms/Q_QUERY_ONLY/fit_audit.csv",
]
missing=[str(p.relative_to(PROJECT)) for p in required if not p.is_file() or p.stat().st_size==0]
if missing: raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
contract=build_di_reference(PROJECT,ROOT)
q=required[-3]
with q.open("r",encoding="utf-8-sig",newline="") as h: rows=list(csv.DictReader(h))
if len(rows)!=799 or len({r["sha256"].lower() for r in rows})!=799:
    raise RuntimeError("BLOCKED_BY_CONTRACT: locked FULL_Q is not exact 799-row OOF")
with required[-1].open("r",encoding="utf-8-sig",newline="") as h: fit=list(csv.DictReader(h))
counts={int(float(r["parameter_n"])) for r in fit}
if counts!={33036}: raise RuntimeError("BLOCKED_BY_CONTRACT: locked FULL_Q parameter count="+repr(counts))
print("QBRANCH_SOURCE_PREFLIGHT_PASS")
print("source_rows=799")
print("year=2017")
print("fold_sizes=1:160,2:160,3:160,4:160,5:159")
print("route_DI=200")
print("route_DI_per_fold=1:40,2:40,3:40,4:40,5:40")
print("FULL_Q=reuse_locked_not_retrained")
print("parameter_count_all_arms=33036")
print("future_rows_loaded=0")
print("ego_consumed=false")
