#!/usr/bin/env python3
from __future__ import annotations
import csv,json,shutil
from pathlib import Path

ROOT=Path("artifacts/care-di25-c43-routed-training-source-v1");BASE=Path("artifacts/care-di25-compact43-relationquery-source-v1")
ROUTER=Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv")
REQ=(ROUTER,BASE/"cache/compact43_relation_source.npz",BASE/"preflight/di25_reference/source_predictions.csv",BASE/"arms/C43_RELATION_QUERY/source_predictions.csv",BASE/"arms/C43_RELATION_QUERY/fit_audit.csv",Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"))
def read(path):
    with path.open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def main():
    missing=[str(p) for p in REQ if not p.is_file() or p.stat().st_size<10]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
    rows=read(ROUTER);folds={f:sum(int(float(r["fold"]))==f for r in rows) for f in range(1,6)};routes={f:sum(int(float(r["fold"]))==f and int(float(r["route_DI"]))==1 for r in rows) for f in range(1,6)}
    fit=read(BASE/"arms/C43_RELATION_QUERY/fit_audit.csv")
    if len(rows)!=799 or len({r["sha256"].lower() for r in rows})!=799 or {int(float(r["year"])) for r in rows}!={2017} or folds!={1:160,2:160,3:160,4:160,5:159} or routes!={1:40,2:40,3:40,4:40,5:40}:raise RuntimeError("BLOCKED_BY_CONTRACT: source/DI25 changed")
    if len(fit)!=15 or {int(float(r["parameter_n"])) for r in fit}!={32292}:raise RuntimeError("BLOCKED_BY_CONTRACT: locked C43-RQ fit audit changed")
    target=ROOT/"preflight/di25_reference";target.mkdir(parents=True,exist_ok=True);shutil.copytree(BASE/"preflight/di25_reference",target,dirs_exist_ok=True)
    contract={"source_rows":799,"year":2017,"fold_sizes":folds,"route_DI":200,"route_DI_per_fold":routes,"outer_train_routed_per_fold":160,"T0_locked_not_retrained":True,"T1":"routed neural loss; all-train backgrounds/scalers","T2":"routed neural loss/backgrounds/scalers","selected_epochs_reused_from_T0":True,"parameter_count":32292,"future_rows_loaded":0,"ego_consumed":False}
    (ROOT/"preflight/contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    print("C43_ROUTED_TRAINING_SOURCE_PREFLIGHT_PASS");print("source_rows=799 year=2017 route_DI=200 outer_train_routed_per_fold=160");print("T0=reuse_locked_not_retrained selected_epochs=reuse_locked_T0");print("future_rows_loaded=0 ego_consumed=false")
if __name__=="__main__":main()

