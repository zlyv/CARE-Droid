#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${CARE_ARG_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
"$PYTHON_BIN" -B scripts/care-api-argument-effects-2017-v1/selftest.py
exec "$PYTHON_BIN" -u -B scripts/care-api-argument-effects-2017-v1/run_experiment.py all --project "$PROJECT_DIR" "$@"
