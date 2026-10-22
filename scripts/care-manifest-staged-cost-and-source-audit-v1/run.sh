#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
: "${MSC_OUT:?new isolated output required}"
export TMPDIR="$MSC_OUT/tmp" XDG_CACHE_HOME="$MSC_OUT/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd /home/boyuanjie/projects/tasa-droid
/home/boyuanjie/envs/tif/bin/python -B scripts/care-manifest-staged-cost-and-source-audit-v1/run.py >> "$MSC_OUT/run.log" 2>&1
