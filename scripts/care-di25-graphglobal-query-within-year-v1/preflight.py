#!/usr/bin/env python3
"""Fail-closed audit for N0/Q same-year OOF without training."""
from __future__ import annotations
import json,sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import RESOLVED,ROUTER,ROOT,ROUTE_COUNTS,YEAR_COUNTS,read_csv,unique

SOURCE=Path("artifacts/care-di25-graphglobal-path-source-v1")

def main():
    required=[
        ROUTER,RESOLVED,
        Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/router_protocol.json"),
        Path("scripts/care-route-stage2-within-year-matrix-v1/run_within_year_arm.py"),
        Path("scripts/care-di25-graphglobal-query-future-v1/graphglobal_path_models.py"),
        Path("artifacts/care-route-stage2-future-matrix-v1/shared/future_interz.npz"),
        Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"),
        Path("artifacts/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"),
        Path("artifacts/care-future-local-assets-v2/regions_2018_2021.npz"),
        Path("artifacts/care-future-local-assets-v2/future_asset_gate.txt"),
        SOURCE/"results/summary.json",SOURCE/"results/route_integrity.csv",
    ]
    for arm in ("N0_NO_GRAPHGLOBAL","Q_QUERY_ONLY"):
        required += [SOURCE/"arms"/arm/"source_predictions.csv",SOURCE/"arms"/arm/"fit_audit.csv"]
    resolved={}
    if RESOLVED.is_file():
        resolved=json.loads(RESOLVED.read_text(encoding="utf-8"))
        for key in ("features","schema","source_manifest"):
            if key in resolved:required.append(Path(resolved[key]))
    missing=sorted(set(str(p) for p in required if not p.is_file() or p.stat().st_size==0))
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty prerequisites:\n- "+"\n- ".join(missing))
    if "passed: true" not in Path("artifacts/care-future-local-assets-v2/future_asset_gate.txt").read_text(encoding="utf-8").casefold():
        raise RuntimeError("BLOCKED_BY_CONTRACT: future asset gate is not PASS")
    for key in ("features","schema","source_manifest"):
        if key not in resolved:raise RuntimeError("BLOCKED_BY_CONTRACT: resolved input missing "+key)
    protocol=json.loads(Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/router_protocol.json").read_text(encoding="utf-8"))
    if protocol.get("protocol")!="strict independent within-year five-fold OOF" or protocol.get("feature_contract_version")!="source-frozen-compact76-foldsafe-v1":
        raise RuntimeError("BLOCKED_BY_CONTRACT: same-year router protocol changed")
    rows=read_csv(ROUTER);unique(rows)
    years=Counter(int(float(r["year"])) for r in rows)
    routes=Counter(int(float(r["year"])) for r in rows if int(float(r["route_DI"]))==1)
    if dict(years)!=YEAR_COUNTS or dict(routes)!=ROUTE_COUNTS:raise RuntimeError(f"cohort/route changed years={dict(years)} routes={dict(routes)}")
    folds={year:Counter() for year in YEAR_COUNTS}
    for row in rows:
        year=int(float(row["year"]));fold=int(float(row["fold"]));folds[year][fold]+=1
        if int(float(row["label"])) not in (0,1) or not 0<float(row["p_stage1"])<1:raise RuntimeError("invalid label/base probability")
    if any(set(v)!=set(range(1,6)) for v in folds.values()):raise RuntimeError("incomplete five-fold coverage")
    source=json.loads((SOURCE/"results/summary.json").read_text(encoding="utf-8"))
    if source.get("source_rows")!=799 or source.get("route_DI")!=200 or source.get("route_difference_count")!=0 or source.get("ego_consumed") is not False:raise RuntimeError("source N0/Q contract changed")
    for arm in ("N0_NO_GRAPHGLOBAL","Q_QUERY_ONLY"):
        params={int(float(r["parameter_n"])) for r in read_csv(SOURCE/"arms"/arm/"fit_audit.csv")}
        if params!={33036}:raise RuntimeError(f"{arm} parameter count changed {params}")
    with np.load("artifacts/care-future-local-assets-v2/regions_2018_2021.npz",allow_pickle=False) as z:
        ids=z["sha256"].astype(str)
        if len(ids)!=1989 or len(set(ids))!=1989 or not {"host_values","host_offsets","real_host_count"}.issubset(z.files):raise RuntimeError("Host-only region asset contract failed")
    ROOT.mkdir(parents=True,exist_ok=True)
    report={"passed":True,"protocol":"independent same-year five-fold OOF","years":YEAR_COUNTS,"DI25_routes":ROUTE_COUNTS,"parameter_count":33036,"ego_consumed":False,"temporal_generalization_result":False}
    (ROOT/"preflight.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("GRAPHGLOBAL_QUERY_WITHIN_YEAR_PREFLIGHT_PASS")
    print("same_year_rows=1989 year_counts=2018:793,2019:399,2020:397,2021:400")
    print("route_DI=497 route_counts=2018:198,2019:100,2020:99,2021:100")
    print("folds=independent 5-fold per year")
    print("parameter_count=33036")
    print("ego_consumed=false")

if __name__=="__main__":main()
