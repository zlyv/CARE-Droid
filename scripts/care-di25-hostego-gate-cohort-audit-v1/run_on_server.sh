#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
/home/boyuanjie/envs/tif/bin/python scripts/care-di25-hostego-gate-cohort-audit-v1/preflight.py
/home/boyuanjie/envs/tif/bin/python scripts/care-di25-hostego-gate-cohort-audit-v1/run_gate_cohort_audit.py
cat artifacts/care-di25-hostego-dmd-bounded-v1/gate_cohort_audit/HISTORICAL_GATE_COHORT_REPORT.txt
