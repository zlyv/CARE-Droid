#!/usr/bin/env python3
from pathlib import Path
required=[
 Path("scripts/care-di25-hostego-within-year-v1/run_m1_within_year.py"),
 Path("scripts/care-di25-hostego-stage2-mechanism-v1/instrumented_model.py"),
 Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv"),
 Path("artifacts/care-future-local-assets-v2/host_regions_2018_2021.npz"),
 Path("artifacts/care-future-local-assets-v2/ego_1hop_regions_2018_2021.npz"),
 Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/instrumented_rerun/apk_internal.csv"),
 Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/apk_stage2_internal.csv"),
 Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/host_internal_diagnostics.csv"),
 Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/routed_unfixed_failure_types.csv"),
]
missing=[str(p) for p in required if not p.is_file()]
if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing:\n- "+"\n- ".join(missing))
print("FINAL_REPLAY_PREFLIGHT_PASS")
