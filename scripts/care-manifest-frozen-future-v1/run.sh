#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
: "${MFF_OUT:?Set a fresh isolated run directory}"
export TMPDIR="$MFF_OUT/tmp" XDG_CACHE_HOME="$MFF_OUT/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd /home/boyuanjie/projects/tasa-droid
/home/boyuanjie/envs/tif/bin/python -B scripts/care-manifest-frozen-future-v1/run.py "$@" >> "$MFF_OUT/run.log" 2>&1
