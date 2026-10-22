#!/usr/bin/env python3
"""Fail closed before any DMD-BC training begins."""
from pathlib import Path
import json

REQUIRED=[
 "scripts/care-di25-hostego-within-year-v1/run_m1_within_year.py",
 "scripts/care-di25-hostego-within-year-v1/host_ego_models.py",
 "scripts/care-di25-hostego-within-year-v1/within_matrix/run_within_year_arm.py",
 "artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv",
 "artifacts/care-di25-hostego-within-year-v1/m1_hostego_anomaly_bounded/within_year_m1_predictions.csv",
 "artifacts/care-future-local-assets-v2/host_regions_2018_2021.npz",
 "artifacts/care-future-local-assets-v2/ego_1hop_regions_2018_2021.npz",
]
missing=[x for x in REQUIRED if not Path(x).is_file()]
if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: preflight missing:\n- "+"\n- ".join(missing))
root=Path("artifacts/care-di25-hostego-dmd-bounded-v1/preflight");root.mkdir(parents=True,exist_ok=True)
(root/"file_gate.json").write_text(json.dumps({"passed":True,"required":REQUIRED},indent=2)+"\n",encoding="utf-8")
print("DMD_BOUNDED_PREFLIGHT_PASS")

