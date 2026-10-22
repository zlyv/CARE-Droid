#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
PROJECT="${MEEA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
PYTHON="${MEEA_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
cd "$PROJECT"
"$PYTHON" -B scripts/care-multiyear-error-evidence-audit-v1/selftest.py
"$PYTHON" -B scripts/care-multiyear-error-evidence-audit-v1/run_audit.py "$@"
