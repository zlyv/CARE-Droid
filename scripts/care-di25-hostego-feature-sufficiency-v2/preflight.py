#!/usr/bin/env python3
from pathlib import Path

required = [
 Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/instrumented_rerun/checkpoint_manifest.csv"),
 Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/instrumented_rerun/per_seed_apk_internal.csv"),
 Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv"),
 Path("scripts/care-di25-hostego-within-year-v1/run_m1_within_year.py"),
 Path("scripts/care-di25-hostego-within-year-v1/within_matrix/run_within_year_arm.py"),
 Path("scripts/care-di25-hostego-stage2-mechanism-v1/instrumented_model.py"),
 Path("scripts/care-di25-hostego-stage2-mechanism-v1/run_instrumented_m1.py"),
]
missing=[str(p) for p in required if not p.is_file()]
if missing: raise RuntimeError("BLOCKED_BY_CONTRACT: preflight missing:\n- " + "\n- ".join(missing))
forbidden = "optimizer" + ".step("
for path in Path(__file__).resolve().parent.glob("*.py"):
    text=path.read_text(encoding="utf-8")
    if forbidden in text: raise RuntimeError("forbidden neural optimizer call in "+str(path))
print("FEATURE_SUFFICIENCY_V2_PREFLIGHT_PASS")
