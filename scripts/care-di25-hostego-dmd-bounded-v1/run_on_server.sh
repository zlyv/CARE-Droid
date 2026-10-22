#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
ROOT=artifacts/care-di25-hostego-dmd-bounded-v1
mkdir -p "$ROOT"/{preflight,B0,B1_nokeep,B2_full,audits,checkpoints,reports}
"$PY" scripts/care-di25-hostego-dmd-bounded-v1/selftest.py
"$PY" scripts/care-di25-hostego-dmd-bounded-v1/preflight.py
"$PY" scripts/care-di25-hostego-dmd-bounded-v1/run_dmd_experiment.py
"$PY" scripts/care-di25-hostego-dmd-bounded-v1/summarize_dmd.py
cat "$ROOT/reports/FINAL_DMD_BOUNDED_REPORT.txt"

