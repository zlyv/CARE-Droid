#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path

REQUIRED=(
 Path("artifacts/p1-features/features.csv"),
 Path("artifacts/p1-features/feature_schema.json"),
 Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"),
 Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv"),
 Path("scripts/care-multiview-conformal-router-source-v1/run_multiview_conformal_router_source_v1.py"),
)
OUT=Path("artifacts/care-global24-codecall-confirm-source-v1")


def read(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def norm(v):return str(v or "").strip().lower()


def main():
    missing=[str(p) for p in REQUIRED if not p.is_file() or p.stat().st_size<10]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
    rows=read(REQUIRED[3]);ids=[norm(r.get("sha256")) for r in rows]
    needed={"sha256","year","label","fold","p_stage1","route_DI"}
    if not rows or not needed.issubset(rows[0]):raise RuntimeError("BLOCKED_BY_CONTRACT: router columns changed")
    years={int(float(r["year"])) for r in rows};folds=[int(float(r["fold"])) for r in rows]
    route=[int(float(r["route_DI"])) for r in rows]
    fold_sizes={f:folds.count(f) for f in sorted(set(folds))}
    route_by_fold={f:sum(v for v,g in zip(route,folds) if g==f) for f in sorted(set(folds))}
    if len(rows)!=799 or len(set(ids))!=799 or years!={2017} or fold_sizes!={1:160,2:160,3:160,4:160,5:159} or sum(route)!=200 or route_by_fold!={1:40,2:40,3:40,4:40,5:40}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen source/DI25 contract changed")
    OUT.mkdir(parents=True,exist_ok=True)
    audit={"passed":True,"source_rows":799,"unique_sha":799,"year":2017,"fold_sizes":fold_sizes,
           "route_DI":200,"route_DI_per_fold":route_by_fold,"future_rows_loaded":0,
           "stage2_training_invoked":False,"frozen_DI25_modified":False}
    (OUT/"preflight.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print("GLOBAL24_CODECALL_CONFIRM_PREFLIGHT_PASS")
    print("source_rows=799 unique_sha=799 year=2017")
    print("fold_sizes=1:160,2:160,3:160,4:160,5:159")
    print("route_DI=200 route_DI_per_fold=1:40,2:40,3:40,4:40,5:40")
    print("future_rows_loaded=0 stage2_training_invoked=false frozen_DI25_modified=false")


if __name__=="__main__":main()
