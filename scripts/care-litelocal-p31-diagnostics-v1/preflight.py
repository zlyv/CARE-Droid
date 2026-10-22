#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from collections import Counter
from pathlib import Path
import numpy as np

OUT=Path("artifacts/care-litelocal-p31-diagnostics-v1")
SOURCE_FEATURES=Path("artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz")
SOURCE_ROUTER=Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv")
SOURCE_PRED=Path("artifacts/care-litelocal-stage1-attribution-source-v1/source_member_oof_predictions.csv")
WITHIN=Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv")
LINEAGES=(Path("artifacts/care-future-local-assets-v2/future_lineage_2018_2021.csv"),Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/future_lineage_2018_2021.csv"),Path("artifacts/care-di25-noego-safe-authority-full-v1/preflight/future_lineage_2018_2021_exact.csv"))
MAPPINGS=(Path("scripts/care-source-recovery-stage2only-v1/extractor_repo/configs/protocol/semantic_mapping_v1.yaml"),Path("scripts/care-route-stage2-future-matrix-v1/configs/semantic_mapping_v1.yaml"))
EXPECTED={2018:793,2019:399,2020:397,2021:400}

def read(path):
    with path.open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def norm(v):return str(v or "").strip().casefold()
def year(row):return int(float(row.get("metadata_year") or row.get("year")))
def apk_path(row):return Path(row.get("absolute_path") or row.get("apk_path") or row.get("path") or "")

def usable_lineage(path):
    if not path.is_file() or path.stat().st_size<100:return None
    rows=read(path)
    try:
        counts=Counter(year(r) for r in rows);ids=[norm(r.get("sha256") or r.get("sample_id")) for r in rows]
        paths=[apk_path(r) for r in rows]
    except Exception:return None
    if len(rows)!=1989 or len(set(ids))!=1989 or dict(counts)!=EXPECTED or not all(p.is_file() for p in paths):return None
    return rows

def main():
    required=(SOURCE_FEATURES,SOURCE_ROUTER,SOURCE_PRED,WITHIN)
    missing=[str(p) for p in required if not p.is_file() or p.stat().st_size<100]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
    lineage_path=None;lineage=None
    for candidate in LINEAGES:
        rows=usable_lineage(candidate)
        if rows is not None:lineage_path=candidate;lineage=rows;break
    mapping=next((p for p in MAPPINGS if p.is_file() and p.stat().st_size>100),None)
    if lineage is None:raise RuntimeError("BLOCKED_BY_CONTRACT: no complete 1989-row future lineage with APK paths and labels")
    if mapping is None:raise RuntimeError("BLOCKED_BY_CONTRACT: semantic mapping missing")
    source=read(SOURCE_ROUTER);sp=read(SOURCE_PRED);within=read(WITHIN)
    if len(source)!=799 or len({norm(r["sha256"]) for r in source})!=799 or sum(int(float(r["route_DI"])) for r in source)!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: source contract changed")
    if len(sp)!=799 or "p_P31__MEAN_COMMITTEE" not in sp[0]:raise RuntimeError("BLOCKED_BY_CONTRACT: locked P31 member OOF predictions missing")
    within_ids=[norm(r["sha256"]) for r in within];future_ids=[norm(r.get("sha256") or r.get("sample_id")) for r in lineage]
    if len(within)!=1989 or len(set(within_ids))!=1989 or set(within_ids)!=set(future_ids) or Counter(int(float(r["year"])) for r in within)!=Counter(EXPECTED):raise RuntimeError("BLOCKED_BY_CONTRACT: within-year fold/cohort contract changed")
    for target_year in EXPECTED:
        year_rows=[r for r in within if int(float(r["year"]))==target_year]
        if set(int(float(r["fold"])) for r in year_rows)!={1,2,3,4,5}:raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete within-year folds")
    OUT.mkdir(parents=True,exist_ok=True);pre=OUT/"preflight";pre.mkdir(exist_ok=True)
    label_free=[]
    for r in lineage:
        label_free.append({"sha256":norm(r.get("sha256") or r.get("sample_id")),"year":year(r),"absolute_path":str(apk_path(r).resolve())})
    with (pre/"future_lineage_label_free.csv").open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=("sha256","year","absolute_path"));w.writeheader();w.writerows(label_free)
    resolved={"source_features":str(SOURCE_FEATURES.resolve()),"source_router":str(SOURCE_ROUTER.resolve()),"source_predictions":str(SOURCE_PRED.resolve()),"within_router":str(WITHIN.resolve()),"future_label_source_posthoc_only":str(lineage_path.resolve()),"future_label_free":str((pre/"future_lineage_label_free.csv").resolve()),"mapping":str(mapping.resolve()),"source_rows":799,"future_rows":1989,"future_year_counts":EXPECTED,"frozen_future_labels_loaded_by_model":False,"within_year_labels_used_for_training":True}
    (pre/"resolved_inputs.json").write_text(json.dumps(resolved,indent=2)+"\n",encoding="utf-8")
    print("LITELOCAL_P31_DIAGNOSTICS_PREFLIGHT_PASS")
    print("source_rows=799 future_rows=1989 year_counts=2018:793,2019:399,2020:397,2021:400")
    print("candidate=P31_MEAN_COMMITTEE source_BA=0.9261309523809524 source_errors=59")
    print("frozen_future_labels_loaded_by_model=false within_year_labels_used_for_training=true")
    print("callgraph_constructed=false stage2_training_invoked=false")

if __name__=="__main__":main()
