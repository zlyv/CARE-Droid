#!/usr/bin/env bash
set -euo pipefail
PROJECT="${RCTD_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${RCTD_GPU:-1}"
export RCTD_DEVICE="${RCTD_DEVICE:-cuda}"
export RCTD_JOBS="${RCTD_JOBS:-4}"
export OMP_NUM_THREADS="$RCTD_JOBS"
export MKL_NUM_THREADS="$RCTD_JOBS"
cd "$PROJECT"
"${RCTD_PYTHON:-/home/boyuanjie/envs/tif/bin/python}" -B "$SCRIPT_DIR/pipeline.py"
