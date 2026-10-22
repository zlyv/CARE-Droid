#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
PROJECT="${MYD_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
PYTHON="${MYD_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
cd "$PROJECT"
mkdir -p outputs/care_multiyear_development_v1/logs
STAMP=$(date +%Y%m%d-%H%M%S)
"$PYTHON" -B scripts/care-multiyear-development-v1/run_pipeline.py "$@" 2>&1 |
  tee "outputs/care_multiyear_development_v1/logs/run-${STAMP}-$$.log"
