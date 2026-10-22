#!/usr/bin/env python3
from pathlib import Path

required = [
    Path("scripts/care-di25-hostego-within-year-v1/run_m1_within_year.py"),
    Path("scripts/care-di25-hostego-within-year-v1/host_ego_encoder.py"),
    Path("scripts/care-di25-hostego-within-year-v1/within_matrix/run_within_year_arm.py"),
    Path("artifacts/care-di25-hostego-within-year-v1/m1_hostego_anomaly_bounded/within_year_m1_predictions.csv"),
    Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv"),
    Path("artifacts/care-future-local-assets-v2/host_regions_2018_2021.npz"),
    Path("artifacts/care-future-local-assets-v2/ego_1hop_regions_2018_2021.npz"),
]
missing=[str(p) for p in required if not p.is_file()]
if missing: raise RuntimeError("BLOCKED_BY_CONTRACT: missing frozen inputs:\n- "+"\n- ".join(missing))
print("STAGE2_MECHANISM_PREFLIGHT_PASS")
