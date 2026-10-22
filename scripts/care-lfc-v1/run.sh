#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
: "${LFC_OUT:?Set LFC_OUT to a new outputs/care_lfc_v1/run_* directory}"
cd /home/boyuanjie/projects/tasa-droid
if [[ "${1:-}" == "--resume" ]]; then
  /home/boyuanjie/envs/tif/bin/python -B scripts/care-lfc-v1/runner.py --resume 2>&1 | tee -a "$LFC_OUT/run.log"
else
  /home/boyuanjie/envs/tif/bin/python -B scripts/care-lfc-v1/runner.py 2>&1 | tee -a "$LFC_OUT/run.log"
fi
