#!/usr/bin/env bash
# BFA-V1 dispatcher: reads historical frozen assets, writes only outputs/care_background_functional_attribution_v1.
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TMPDIR="$ROOT/outputs/care_background_functional_attribution_v1/tmp"
cd "$ROOT"
exec /home/boyuanjie/envs/tif/bin/python scripts/care-background-functional-attribution-v1/bfa_run.py "$@"
