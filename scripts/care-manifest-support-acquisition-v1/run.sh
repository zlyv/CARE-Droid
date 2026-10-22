#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
: "${MSA_OUT:?new isolated output required}"
export TMPDIR="$MSA_OUT/tmp" XDG_CACHE_HOME="$MSA_OUT/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd /home/boyuanjie/projects/tasa-droid
/home/boyuanjie/envs/tif/bin/python -B scripts/care-manifest-support-acquisition-v1/run.py "$@" >> "$MSA_OUT/run.log" 2>&1
