#!/usr/bin/env bash
# BFA-V2 dispatcher. Reads historical frozen assets; writes only outputs/care_background_complementarity_v2.
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TMPDIR="$ROOT/outputs/care_background_complementarity_v2/tmp"
cd "$ROOT"
exec /home/boyuanjie/envs/tif/bin/python scripts/care-background-complementarity-v2/bfa_v2.py "$@"
