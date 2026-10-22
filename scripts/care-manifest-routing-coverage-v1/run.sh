#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
: "${MRC_OUT:?Set a new isolated run directory}"
export TMPDIR="$MRC_OUT/tmp" XDG_CACHE_HOME="$MRC_OUT/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd /home/boyuanjie/projects/tasa-droid
/home/boyuanjie/envs/tif/bin/python -B scripts/care-manifest-routing-coverage-v1/run.py "$@" >> "$MRC_OUT/run.log" 2>&1
