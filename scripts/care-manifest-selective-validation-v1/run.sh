#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
: "${MSV_OUT:?Set isolated run directory}"
export TMPDIR="$MSV_OUT/tmp" XDG_CACHE_HOME="$MSV_OUT/cache" TORCH_HOME="$MSV_OUT/cache/torch"
export PYTHONPATH="/home/boyuanjie/projects/tasa-droid/src:/home/boyuanjie/projects/tasa-droid/scripts/care-di25-host-gatedgraph-source-v1"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd /home/boyuanjie/projects/tasa-droid
/home/boyuanjie/envs/tif/bin/python -B scripts/care-manifest-selective-validation-v1/run.py "$@" >> "$MSV_OUT/run.log" 2>&1
