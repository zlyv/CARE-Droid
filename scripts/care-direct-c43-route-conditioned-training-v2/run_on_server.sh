#!/usr/bin/env bash
set -euo pipefail
PROJECT="${RCT_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${RCT_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${RCT_GPU:-1}"
export RCT_DEVICE="${RCT_DEVICE:-cuda}"
export RCT_JOBS="${RCT_JOBS:-4}"
export OMP_NUM_THREADS="${RCT_JOBS}"
export MKL_NUM_THREADS="${RCT_JOBS}"
cd "$PROJECT"
"$PYTHON" -B "$SCRIPT_DIR/pipeline.py"
