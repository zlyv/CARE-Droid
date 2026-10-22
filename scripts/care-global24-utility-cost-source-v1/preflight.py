#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path

REQUIRED=(Path("artifacts/p1-features/features.csv"),Path("artifacts/p1-features/feature_schema.json"),
 Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"),
 Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv"),
 Path("scripts/care-multiview-conformal-router-source-v1/run_multiview_conformal_router_source_v1.py"),
 Path("scripts/care-committee-uncertainty-router-v1/committee_uncertainty_core.py"))


def read(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def norm(v):return str(v or "").strip().lower()


def main():
    missing=[str(p) for p in REQUIRED if not p.is_file() or p.stat().st_size<10]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
    rows=read(REQUIRED[3]);ids=[norm(r["sha256"]) for r in rows];folds=[int(float(r["fold"])) for r in rows];routes=[int(float(r["route_DI"])) for r in rows]
    sizes={f:folds.count(f) for f in sorted(set(folds))};routed={f:sum(v for v,g in zip(routes,folds) if g==f) for f in sorted(set(folds))}
    required_cols={"sha256","year","label","fold","p_stage1","route_DI"}
    if len(rows)!=799 or len(set(ids))!=799 or not required_cols.issubset(rows[0]) or {int(float(r["year"])) for r in rows}!={2017} or sizes!={1:160,2:160,3:160,4:160,5:159} or routed!={1:40,2:40,3:40,4:40,5:40}:raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort/DI25 changed")
    Path("artifacts/care-global24-utility-cost-source-v1").mkdir(parents=True,exist_ok=True)
    Path("artifacts/care-global24-utility-cost-source-v1/preflight.json").write_text(json.dumps({"passed":True,"source_rows":799,"year":2017,"fold_sizes":sizes,"route_DI":sum(routes),"route_DI_per_fold":routed,"future_rows_loaded":0,"stage2_training_invoked":False},indent=2)+"\n")
    print("GLOBAL24_UTILITY_COST_PREFLIGHT_PASS")
    print("source_rows=799 year=2017 fold_sizes=1:160,2:160,3:160,4:160,5:159")
    print("route_DI=200 route_DI_per_fold=1:40,2:40,3:40,4:40,5:40")
    print("future_rows_loaded=0 stage2_training_invoked=false")


if __name__=="__main__":main()
