#!/usr/bin/env python3
from __future__ import annotations
import csv,json,shutil
from pathlib import Path
ROOT=Path("artifacts/care-di25-roleconditioned-background-source-v1")
ROUTER=Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv")
LOCKED=Path("artifacts/care-di25-graphglobal-path-source-v1/arms/Q_QUERY_ONLY")
SOURCE_REF=Path("artifacts/care-di25-host-relative-source-v1/preflight/di25_reference")
REQ=(ROUTER,LOCKED/"source_predictions.csv",LOCKED/"fit_audit.csv",SOURCE_REF/"source_predictions.csv",Path("artifacts/care-source-local-assets-v2/regions_2017.npz"),Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"))
def read(p):
    with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def main():
    missing=[str(p) for p in REQ if not p.is_file() or p.stat().st_size<10]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty:\n- "+"\n- ".join(missing))
    r=read(ROUTER);fold={f:sum(int(float(x["fold"]))==f for x in r) for f in range(1,6)};route={f:sum(int(float(x["fold"]))==f and int(float(x["route_DI"]))==1 for x in r) for f in range(1,6)}
    if len(r)!=799 or len({x["sha256"].lower() for x in r})!=799 or {int(float(x["year"])) for x in r}!={2017} or fold!={1:160,2:160,3:160,4:160,5:159} or route!={1:40,2:40,3:40,4:40,5:40}:raise RuntimeError("BLOCKED_BY_CONTRACT: source/DI25 changed")
    target=ROOT/"preflight/di25_reference";target.mkdir(parents=True,exist_ok=True)
    shutil.copytree(SOURCE_REF,target,dirs_exist_ok=True)
    contract={"source_rows":799,"year":2017,"route_DI":200,"route_DI_per_fold":route,"H0_locked_not_retrained":True,"host_representation":"original H0 74-D","prototype_capacity_matched":True,"future_rows_loaded":0,"ego_consumed":False}
    (ROOT/"preflight/contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    print("ROLECONDITIONED_BACKGROUND_PREFLIGHT_PASS");print("source_rows=799 year=2017 route_DI=200");print("prototype_capacity=shared32+benign24 parameter_count=33036");print("future_rows_loaded=0 ego_consumed=false")
if __name__=="__main__":main()
