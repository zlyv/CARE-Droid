#!/usr/bin/env python3
"""Static and real-asset preflight for the strict 2017 ViewDI audit."""
from __future__ import annotations
import ast,csv,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
required=[
    PROJECT/"artifacts/p1-features/features.csv",
    PROJECT/"artifacts/p1-features/feature_schema.json",
    PROJECT/"artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv",
    PROJECT/"artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv",
    PROJECT/"scripts/care-multiview-conformal-router-source-v1/run_multiview_conformal_router_source_v1.py",
    PROJECT/"scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py",
]
missing=[str(p.relative_to(PROJECT)) for p in required if not p.is_file() or p.stat().st_size==0]
if missing: raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty:\n- "+"\n- ".join(missing))
for name in ("run_viewdi_source.py","viewdi_core.py"):
    ast.parse((ROOT/name).read_text(encoding="utf-8"),filename=name)
with required[3].open("r",encoding="utf-8-sig",newline="") as h: rows=list(csv.DictReader(h))
ids=[r["sha256"].strip().lower() for r in rows];folds=[int(float(r["fold"])) for r in rows]
years={int(float(r["year"])) for r in rows};routes=[int(float(r["route_DI"])) for r in rows]
sizes={f:folds.count(f) for f in sorted(set(folds))};byfold={f:sum(v for v,g in zip(routes,folds) if g==f) for f in sorted(set(folds))}
if len(rows)!=799 or len(set(ids))!=799 or years!={2017} or sizes!={1:160,2:160,3:160,4:160,5:159} or byfold!={1:40,2:40,3:40,4:40,5:40}:
    raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort/DI25 changed")
print("VIEWDI_SOURCE_PREFLIGHT_PASS")
print("source_rows=799")
print("year=2017")
print("route_DI=200")
print("future_rows_loaded=0")
print("stage2_training_invoked=false")

