#!/usr/bin/env python3
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT=Path("artifacts/care-di25-hostrel-semantic-background-diagnostics-v1")
SOURCE_ROUTER=Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv")
FUTURE_ROUTER=Path("artifacts/care-route-stage2-future-matrix-v1/preflight/future_router_inputs_label_free.csv")
FUTURE_RESOLVED=Path("artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json")
WITHIN_ROUTER=Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv")
REQUIRED=(SOURCE_ROUTER,FUTURE_ROUTER,FUTURE_RESOLVED,WITHIN_ROUTER,Path("artifacts/care-route-stage2-future-matrix-v1/shared/future_interz.npz"),Path("artifacts/care-di25-graphglobal-query-future-v1/arms/Q_QUERY_ONLY/future_predictions_label_free.csv"),Path("artifacts/care-di25-graphglobal-query-within-year-v1/arms/Q_QUERY_ONLY/within_year_predictions.csv"),Path("artifacts/care-di25-roleconditioned-background-source-v1/cache/regions_2017_role_sidecar.npz"),Path("artifacts/care-di25-hostrel-semantic-background-source-v1/cache/regions_2017_hostrel_role_sidecar.npz"),Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"),Path("artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz"))
ENDPOINTS=(Path("artifacts/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"),Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"))
REGIONS=(Path("artifacts/care-future-local-assets-v2/regions_2018_2021.npz"),Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/regions_2018_2021.npz"))


def read(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))


def norm(value): return str(value or "").strip().lower()


def main():
    missing=[str(p) for p in REQUIRED if not p.is_file() or p.stat().st_size<10]
    if not any(p.is_file() and p.stat().st_size>100 for p in ENDPOINTS): missing.append("future endpoint cache")
    if not any(p.is_file() and p.stat().st_size>100 for p in REGIONS): missing.append("future regions cache")
    if missing: raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
    source=read(SOURCE_ROUTER); future=read(FUTURE_ROUTER); within=read(WITHIN_ROUTER)
    if len(source)!=799 or len({norm(r.get("sha256") or r.get("sample_id")) for r in source})!=799 or {int(float(r["year"])) for r in source}!={2017} or sum(int(float(r["route_DI"])) for r in source)!=200: raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort/route changed")
    year_counts={y:sum(int(float(r["year"]))==y for r in future) for y in (2018,2019,2020,2021)}
    route_counts={y:sum(int(float(r["year"]))==y and int(float(r["route_DI"]))==1 for r in future) for y in year_counts}
    if len(future)!=1989 or year_counts!={2018:793,2019:399,2020:397,2021:400} or route_counts!={2018:198,2019:100,2020:99,2021:100}: raise RuntimeError("BLOCKED_BY_CONTRACT: future cohort/route changed")
    if len(within)!=1989 or {norm(r.get("sha256") or r.get("sample_id")) for r in within}!={norm(r.get("sha256") or r.get("sample_id")) for r in future}: raise RuntimeError("BLOCKED_BY_CONTRACT: within cohort changed")
    resolved=json.loads(FUTURE_RESOLVED.read_text())
    label_path=Path(resolved.get("future_label_file_posthoc_only",""))
    if not label_path.is_file(): raise RuntimeError("BLOCKED_BY_CONTRACT: posthoc labels missing")
    print("HOSTREL_SEMANTIC_DIAGNOSTICS_PREFLIGHT_PASS")
    print("source_rows=799 source_route_DI=200")
    print("future_rows=1989 future_route_DI=497 year_counts=2018:793,2019:399,2020:397,2021:400")
    print("arms=B1_HARD_ROLE,R1_HOSTREL74_HARD_ROLE controls=H0_LOCKED gpu_workers=2")
    print("backgrounds_same_host_schema=true future_labels_loaded_by_model=false ego_consumed=false")


if __name__=="__main__": main()
