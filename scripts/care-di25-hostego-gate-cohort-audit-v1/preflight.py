#!/usr/bin/env python3
from pathlib import Path
required=[Path("artifacts/care-di25-hostego-dmd-bounded-v1/r4_component_audit/component_probe_oof_predictions.csv"),
 Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/apk_stage2_internal.csv"),
 Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/routed_unfixed_failure_types.csv")]
missing=[str(p) for p in required if not p.is_file()]
if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing inputs:\n- "+"\n- ".join(missing))
print("HISTORICAL_GATE_COHORT_PREFLIGHT_PASS")
