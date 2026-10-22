#!/usr/bin/env python3
"""Fail-closed preflight without opening future labels."""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import FUTURE_INPUT, RESOLVED, ROUTE_COUNTS, SOURCE_ROOT, YEAR_COUNTS, norm, read_csv, unique

EXPECTED={
    "N0_NO_GRAPHGLOBAL":{"BA":0.911093,"AUC":0.943966,"F1":0.907672,"fixed":82,"broken":9,"net":73},
    "Q_QUERY_ONLY":{"BA":0.917359,"AUC":0.944192,"F1":0.914729,"fixed":87,"broken":9,"net":78},
}

def main():
    required=[
        RESOLVED, FUTURE_INPUT,
        Path("scripts/care-route-stage2-future-matrix-v1/run_future_arm.py"),
        Path("scripts/care-route-stage2-future-matrix-v1/vendor/run_s7_interz_future_diagnostic.py"),
        Path("scripts/care-di25-graphglobal-query-future-v1/graphglobal_path_models.py"),
        SOURCE_ROOT/"results/summary.json", SOURCE_ROOT/"results/route_integrity.csv",
        SOURCE_ROOT/"results/REPORT.txt",
        Path("artifacts/care-source-local-assets-v2/regions_2017.npz"),
        Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"),
        Path("artifacts/care-source-local-assets-v2/source_asset_gate.txt"),
        Path("artifacts/care-future-local-assets-v2/regions_2018_2021.npz"),
        Path("artifacts/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"),
        Path("artifacts/care-future-local-assets-v2/future_asset_gate.txt"),
        Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"),
        Path("artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz"),
    ]
    for arm in EXPECTED:
        required += [SOURCE_ROOT/"arms"/arm/"source_summary.csv", SOURCE_ROOT/"arms"/arm/"source_predictions.csv", SOURCE_ROOT/"arms"/arm/"fit_audit.csv"]
    resolved={}
    if RESOLVED.is_file() and RESOLVED.stat().st_size>0:
        resolved=json.loads(RESOLVED.read_text(encoding="utf-8"))
        for key in ("features","schema","source_scores","future_router_inputs_label_free","future_label_file_posthoc_only"):
            if key in resolved: required.append(Path(resolved[key]))
    missing=sorted(set(str(p) for p in required if not p.is_file() or p.stat().st_size==0))
    if missing: raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
    for gate in (Path("artifacts/care-source-local-assets-v2/source_asset_gate.txt"),Path("artifacts/care-future-local-assets-v2/future_asset_gate.txt")):
        if "passed: true" not in gate.read_text(encoding="utf-8").casefold():
            raise RuntimeError("BLOCKED_BY_CONTRACT: asset gate is not PASS: "+str(gate))

    contract=json.loads((HERE/"freeze_contract.json").read_text(encoding="utf-8"))
    if not contract.get("human_freeze_approved") or contract.get("candidate")!="Q_QUERY_ONLY":
        raise RuntimeError("BLOCKED_BY_CONTRACT: Q-only is not frozen")
    source=json.loads((SOURCE_ROOT/"results/summary.json").read_text(encoding="utf-8"))
    if source.get("source_rows")!=799 or source.get("route_DI")!=200 or source.get("route_difference_count")!=0 or source.get("future_rows_loaded")!=0 or source.get("ego_consumed") is not False:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source hard audit changed")
    route_rows=read_csv(SOURCE_ROOT/"results/route_integrity.csv")
    relevant=[r for r in route_rows if r.get("arm") in EXPECTED]
    if len(relevant)!=1598 or any(int(float(r["different"])) for r in relevant):
        raise RuntimeError("BLOCKED_BY_CONTRACT: N0/Q source route identity failed")

    for arm,want in EXPECTED.items():
        summaries=read_csv(SOURCE_ROOT/"arms"/arm/"source_summary.csv")
        rows=[r for r in summaries if arm in r.get("method","") and r.get("method","").endswith("__Bounded")]
        if len(rows)!=1: raise RuntimeError("BLOCKED_BY_CONTRACT: missing source summary "+arm)
        row=rows[0]
        for key,value in want.items():
            observed=float(row[key])
            tolerance=5e-6 if key in {"BA","AUC","F1"} else 0.0
            if abs(observed-value)>tolerance: raise RuntimeError(f"BLOCKED_BY_CONTRACT: {arm} {key} changed {observed} != {value}")
        predictions=read_csv(SOURCE_ROOT/"arms"/arm/"source_predictions.csv")
        unique(predictions,799)
        parameters={int(float(r["parameter_n"])) for r in read_csv(SOURCE_ROOT/"arms"/arm/"fit_audit.csv")}
        if parameters!={33036}: raise RuntimeError(f"BLOCKED_BY_CONTRACT: {arm} parameter count changed {parameters}")

    for key in ("features","schema","source_scores","future_router_inputs_label_free","future_label_file_posthoc_only"):
        if key not in resolved: raise RuntimeError("BLOCKED_BY_CONTRACT: resolved manifest missing "+key)
    future=read_csv(FUTURE_INPUT); unique(future,1989)
    forbidden=[k for k in future[0] if "label" in k.casefold()]
    if forbidden: raise RuntimeError("BLOCKED_BY_CONTRACT: future router input contains label columns "+repr(forbidden))
    years=Counter(int(float(r["year"])) for r in future)
    routes=Counter(int(float(r["year"])) for r in future if int(float(r["route_DI"]))==1)
    if dict(years)!=YEAR_COUNTS or dict(routes)!=ROUTE_COUNTS:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: future cohort/route changed years={dict(years)} routes={dict(routes)}")
    if any(not (0.0<float(r["p_stage1"])<1.0) for r in future):
        raise RuntimeError("BLOCKED_BY_CONTRACT: invalid label-free Stage-I probabilities")
    inter=Path("artifacts/care-route-stage2-future-matrix-v1/shared/future_interz.npz")
    inter_status="reuse_existing" if inter.is_file() and inter.stat().st_size>0 else "build_label_free_before_training"
    print("GRAPHGLOBAL_QUERY_FUTURE_PREFLIGHT_PASS")
    print("source_rows=799 source_route_DI=200 route_difference_count=0")
    print("candidate=Q_QUERY_ONLY control=N0_NO_GRAPHGLOBAL parameter_count=33036")
    print("future_rows=1989 year_counts=2018:793,2019:399,2020:397,2021:400")
    print("future_route_DI=497 route_counts=2018:198,2019:100,2020:99,2021:100")
    print("future_rows_loaded=0")
    print("future_labels_loaded=0")
    print("ego_consumed=false")
    print("future_inter_cache="+inter_status)

if __name__=="__main__": main()
