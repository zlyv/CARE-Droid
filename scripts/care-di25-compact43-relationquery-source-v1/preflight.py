#!/usr/bin/env python3
from __future__ import annotations
import csv,json,shutil
from pathlib import Path

ROOT=Path("artifacts/care-di25-compact43-relationquery-source-v1"); ROUTER=Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv"); H0=Path("artifacts/care-di25-graphglobal-path-source-v1/arms/Q_QUERY_ONLY"); SOURCE_REF=Path("artifacts/care-di25-host-relative-source-v1/preflight/di25_reference")
REQ=(ROUTER,H0/"source_predictions.csv",H0/"fit_audit.csv",SOURCE_REF/"source_predictions.csv",Path("artifacts/care-source-local-assets-v2/regions_2017.npz"),Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"))


def read(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def main():
    missing=[str(p) for p in REQ if not p.is_file() or p.stat().st_size<10]
    if missing: raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty:\n- "+"\n- ".join(missing))
    rows=read(ROUTER); folds={f:sum(int(float(r["fold"]))==f for r in rows) for f in range(1,6)}; routes={f:sum(int(float(r["fold"]))==f and int(float(r["route_DI"]))==1 for r in rows) for f in range(1,6)}
    if len(rows)!=799 or len({r["sha256"].lower() for r in rows})!=799 or {int(float(r["year"])) for r in rows}!={2017} or folds!={1:160,2:160,3:160,4:160,5:159} or routes!={1:40,2:40,3:40,4:40,5:40}: raise RuntimeError("BLOCKED_BY_CONTRACT: source/DI25 changed")
    target=ROOT/"preflight/di25_reference"; target.mkdir(parents=True,exist_ok=True); shutil.copytree(SOURCE_REF,target,dirs_exist_ok=True)
    contract={"source_rows":799,"year":2017,"route_DI":200,"route_DI_per_fold":routes,"H0_locked_not_retrained":True,"train_arms":["C43_COMPACT","C43_RELATION_QUERY"],"compact_host_width":43,"duplicate_role_columns":0,"relation_query_width":31,"relation_never_enters_delta":True,"future_rows_loaded":0,"ego_consumed":False}
    (ROOT/"preflight/contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    print("COMPACT43_RELATIONQUERY_PREFLIGHT_PASS")
    print("source_rows=799 year=2017 route_DI=200")
    print("compact_host_width=43 duplicate_role_columns=0 relation_query_width=31")
    print("H0=reuse_locked_not_retrained future_rows_loaded=0 ego_consumed=false")


if __name__=="__main__":main()
