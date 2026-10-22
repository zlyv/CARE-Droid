#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${CARE_ML_PYTHON:-/home/boyuanjie/envs/tif/bin/python}";S="scripts/care-di25-host-information-loss-audit-v1"
"$PY" "$S/selftest.py";"$PY" "$S/preflight.py";OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 5 "$PY" "$S/run_audit.py"
