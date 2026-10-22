#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
ROOT=/home/boyuanjie/projects/tasa-droid
OUT="$ROOT/outputs/care_lite_stage_v1"
mkdir -p "$OUT"
cd "$ROOT"
if [[ "${1:-}" == "--resume" ]]; then
  /home/boyuanjie/envs/tif/bin/python -B scripts/care-lite-stage-v1/runner.py --resume 2>&1 | tee -a "$OUT/run.log"
else
  /home/boyuanjie/envs/tif/bin/python -B scripts/care-lite-stage-v1/runner.py 2>&1 | tee -a "$OUT/run.log"
fi
