#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path

ROOT=Path("artifacts/care-di25-compact43-relationquery-diagnostics-v1")
SOURCE_ROOT=Path("artifacts/care-di25-compact43-relationquery-source-v1")
SOURCE_ROUTER=Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv")
FUTURE_ROUTER=Path("artifacts/care-route-stage2-future-matrix-v1/preflight/future_router_inputs_label_free.csv")
WITHIN_ROUTER=Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv")
RESOLVED=Path("artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json")
REQUIRED=(SOURCE_ROUTER,FUTURE_ROUTER,WITHIN_ROUTER,RESOLVED,
          Path("artifacts/care-route-stage2-future-matrix-v1/shared/future_interz.npz"),
          SOURCE_ROOT/"cache/compact43_relation_source.npz",
          SOURCE_ROOT/"arms/C43_RELATION_QUERY/contract.json",
          SOURCE_ROOT/"arms/C43_RELATION_QUERY/source_predictions.csv",
          Path("artifacts/care-di25-graphglobal-query-future-v1/arms/Q_QUERY_ONLY/future_predictions_label_free.csv"),
          Path("artifacts/care-di25-graphglobal-query-within-year-v1/arms/Q_QUERY_ONLY/within_year_predictions.csv"),
          Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"),
          Path("artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz"))
REGIONS=(Path("artifacts/care-future-local-assets-v2/regions_2018_2021.npz"),Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/regions_2018_2021.npz"))
ENDPOINTS=(Path("artifacts/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"),Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"))


def read(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def norm(value):return str(value or "").strip().lower()


def main():
    missing=[str(p) for p in REQUIRED if not p.is_file() or p.stat().st_size<10]
    if not any(p.is_file() and p.stat().st_size>100 for p in REGIONS):missing.append("future regions")
    if not any(p.is_file() and p.stat().st_size>100 for p in ENDPOINTS):missing.append("future endpoint cache")
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
    contract=json.loads((SOURCE_ROOT/"arms/C43_RELATION_QUERY/contract.json").read_text())
    checks={"arm":contract.get("arm")=="C43_RELATION_QUERY","parameters":int(contract.get("parameter_count",-1))==32292,
            "compact43":int(contract.get("compact_host_width",-1))==43,"no_duplicate":int(contract.get("duplicate_role_columns",-1))==0,
            "relation_attention":contract.get("relation_enters_attention") is True,"relation_not_delta":contract.get("relation_enters_delta") is False,
            "no_ego":contract.get("ego_consumed") is False}
    if not all(checks.values()):raise RuntimeError("BLOCKED_BY_CONTRACT: source candidate contract changed "+json.dumps(checks))
    source=read(SOURCE_ROUTER);future=read(FUTURE_ROUTER);within=read(WITHIN_ROUTER)
    sids={norm(r.get("sha256") or r.get("sample_id")) for r in source};fids={norm(r.get("sha256") or r.get("sample_id")) for r in future}
    if len(source)!=799 or len(sids)!=799 or {int(float(r["year"])) for r in source}!={2017} or sum(int(float(r["route_DI"])) for r in source)!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort/route changed")
    yc={y:sum(int(float(r["year"]))==y for r in future) for y in (2018,2019,2020,2021)};rc={y:sum(int(float(r["year"]))==y and int(float(r["route_DI"]))==1 for r in future) for y in yc}
    if len(future)!=1989 or len(fids)!=1989 or yc!={2018:793,2019:399,2020:397,2021:400} or rc!={2018:198,2019:100,2020:99,2021:100}:raise RuntimeError("BLOCKED_BY_CONTRACT: future cohort/route changed")
    if len(within)!=1989 or {norm(r.get("sha256") or r.get("sample_id")) for r in within}!=fids:raise RuntimeError("BLOCKED_BY_CONTRACT: within cohort changed")
    resolved=json.loads(RESOLVED.read_text());label_path=Path(resolved.get("future_label_file_posthoc_only",""))
    if not label_path.is_file():raise RuntimeError("BLOCKED_BY_CONTRACT: posthoc future labels missing")
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/"preflight.json").write_text(json.dumps({"passed":True,"source_rows":799,"future_rows":1989,"future_year_counts":yc,"future_DI25_routes":rc,"candidate":"C43_RELATION_QUERY","control":"H0_LOCKED","parameter_count":32292,"future_labels_loaded_by_model":False,"ego_consumed":False},indent=2)+"\n")
    print("COMPACT43_RELATIONQUERY_DIAGNOSTICS_PREFLIGHT_PASS")
    print("source_rows=799 source_route_DI=200")
    print("future_rows=1989 future_route_DI=497 year_counts=2018:793,2019:399,2020:397,2021:400")
    print("candidate=C43_RELATION_QUERY control=H0_LOCKED parameter_count=32292 gpu_workers=2")
    print("relation_enters_attention=true relation_enters_delta=false future_labels_loaded_by_model=false ego_consumed=false")


if __name__=="__main__":main()
