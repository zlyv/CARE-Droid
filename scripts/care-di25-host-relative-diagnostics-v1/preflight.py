#!/usr/bin/env python3
from __future__ import annotations
import csv,json,sys
from pathlib import Path
ROOT=Path("artifacts/care-di25-host-relative-diagnostics-v1")
SOURCE=Path("artifacts/care-di25-host-relative-source-v1")
ARMS=("HM_R","HC_R","HMC_R")
SOURCE_ROUTER=Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv")
FUTURE_ROUTER=Path("artifacts/care-route-stage2-future-matrix-v1/preflight/future_router_inputs_label_free.csv")
FUTURE_RESOLVED=Path("artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json")
WITHIN_ROUTER=Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv")
FUTURE_ENDPOINT_CANDIDATES=(
 Path("artifacts/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"),
 Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"),
)
REQUIRED=(
 SOURCE_ROUTER,
 FUTURE_ROUTER,
 FUTURE_RESOLVED,
 Path("artifacts/care-route-stage2-future-matrix-v1/shared/future_interz.npz"),
 Path("artifacts/care-di25-graphglobal-query-future-v1/arms/Q_QUERY_ONLY/future_predictions_label_free.csv"),
 Path("artifacts/care-di25-graphglobal-query-within-year-v1/arms/Q_QUERY_ONLY/within_year_predictions.csv"),
 WITHIN_ROUTER,
 Path("artifacts/care-route-stage2-within-year-matrix-v1/preflight/resolved_inputs.json"),
 Path("artifacts/care-source-local-assets-v2/regions_2017.npz"),
 Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"),
 Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"),
 Path("artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz"),
)
def read(p):
    with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def norm(x):return str(x or "").strip().lower()
def main():
    missing=[str(p) for p in REQUIRED if not p.is_file() or p.stat().st_size<10]
    if not any(p.is_file() and p.stat().st_size>100 for p in FUTURE_ENDPOINT_CANDIDATES):missing.append("future endpoint cache (artifacts or HDD)")
    missing += [str(SOURCE/"caches"/(a+"_regions_2017.npz")) for a in ARMS if not (SOURCE/"caches"/(a+"_regions_2017.npz")).is_file()]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
    src=read(SOURCE_ROUTER);future=read(FUTURE_ROUTER);within=read(WITHIN_ROUTER)
    sids=[norm(r.get("sha256") or r.get("sample_id")) for r in src];fids=[norm(r.get("sha256") or r.get("sample_id")) for r in future]
    if len(src)!=799 or len(set(sids))!=799 or {int(float(r["year"])) for r in src}!={2017}:raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort changed")
    if sum(int(float(r["route_DI"])) for r in src)!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: source DI25 changed")
    yc={y:sum(int(float(r["year"]))==y for r in future) for y in (2018,2019,2020,2021)}
    rc={y:sum(int(float(r["year"]))==y and int(float(r["route_DI"]))==1 for r in future) for y in yc}
    if len(future)!=1989 or len(set(fids))!=1989 or yc!={2018:793,2019:399,2020:397,2021:400} or rc!={2018:198,2019:100,2020:99,2021:100}:raise RuntimeError("BLOCKED_BY_CONTRACT: future cohort/router changed")
    if len(within)!=1989 or {norm(r.get("sha256") or r.get("sample_id")) for r in within}!=set(fids):raise RuntimeError("BLOCKED_BY_CONTRACT: within-year cohort changed")
    resolved=json.loads(FUTURE_RESOLVED.read_text(encoding="utf-8"))
    for key in ("features","schema","source_manifest","source_scores","future_router_inputs_label_free"):
        value=Path(resolved.get(key,""))
        if not value.is_file() or value.stat().st_size<10:raise RuntimeError("BLOCKED_BY_CONTRACT: resolved input missing "+key+"="+str(value))
    label_path=Path(resolved.get("future_label_file_posthoc_only", ""))
    if not label_path.is_file():raise RuntimeError("BLOCKED_BY_CONTRACT: posthoc label file missing")
    print("HOST_RELATIVE_DIAGNOSTICS_PREFLIGHT_PASS")
    print("source_rows=799 source_route_DI=200")
    print("future_rows=1989 future_route_DI=497")
    print("future_year_counts=2018:793,2019:399,2020:397,2021:400")
    print("future_route_counts=2018:198,2019:100,2020:99,2021:100")
    print("H0=reuse_locked_not_retrained")
    print("three_backgrounds_same_host_schema=true")
    print("gpu_workers=2")
    print("ego_consumed=false")
if __name__=="__main__":main()
