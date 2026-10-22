#!/usr/bin/env python3
from pathlib import Path

REQUIRED = [
    Path("artifacts/care-di25-hostego-dmd-bounded-v1/r4_component_audit/component_probe_oof_predictions.csv"),
    Path("artifacts/care-di25-hostego-dmd-bounded-v1/gate_cohort_audit/historical_cohort_gate_scores.csv"),
    Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/apk_stage2_internal.csv"),
    Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/routed_unfixed_failure_types.csv"),
]
missing = [str(p) for p in REQUIRED if not p.is_file()]
if missing:
    raise RuntimeError("BLOCKED_BY_CONTRACT: missing read-only inputs:\n- " + "\n- ".join(missing))
print("FAILURE_BLOCK_AUDIT_PREFLIGHT_PASS")
