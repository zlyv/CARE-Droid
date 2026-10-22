#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
: "${MCA_OUT:?new isolated output required}"
export TMPDIR="$MCA_OUT/tmp" XDG_CACHE_HOME="$MCA_OUT/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd /home/boyuanjie/projects/tasa-droid
/home/boyuanjie/envs/tif/bin/python -B scripts/care-manifest-acquisition-cascade-audit-v1/run.py "$@" >> "$MCA_OUT/run.log" 2>&1
