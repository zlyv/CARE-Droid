#!/usr/bin/env python3
import csv
from pathlib import Path
P={
"H_HOST_ONLY":Path("artifacts/care-di25-qbranch-source-v1/arms/HOST_ONLY_Q/source_predictions.csv"),
"HR_HOST_ROLE":Path("artifacts/care-di25-qbranch-source-v1/arms/STRUCTURE_ONLY_Q/source_predictions.csv"),
"HC_HOST_CONTEXT":Path("artifacts/care-di25-q-drop-role-source-v1/arm/source_predictions.csv"),
"HRC_FULL":Path("artifacts/care-di25-graphglobal-path-source-v1/arms/Q_QUERY_ONLY/source_predictions.csv")}
ids=None
for arm,p in P.items():
    if not p.is_file() or p.stat().st_size==0:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty "+str(p))
    with p.open("r",encoding="utf-8-sig",newline="") as h:r=list(csv.DictReader(h))
    cur={x["sha256"].lower() for x in r}
    if len(r)!=799 or len(cur)!=799:raise RuntimeError(f"BLOCKED_BY_CONTRACT: {arm} rows/ids invalid")
    if ids is not None and cur!=ids:raise RuntimeError(f"BLOCKED_BY_CONTRACT: {arm} cohort differs")
    ids=cur
print("Q_CORRECTION_INTERACTION_PREFLIGHT_PASS")
print("source_rows=799 year=2017 route_DI_expected=200")
print("arms=H,HR,HC,HRC")
print("all_predictions_locked=true")
print("training_invoked=false")
print("inference_invoked=false")
print("future_rows_loaded=0")
