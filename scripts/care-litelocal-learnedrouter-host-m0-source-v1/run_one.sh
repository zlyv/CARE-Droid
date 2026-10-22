#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
CODE="scripts/care-litelocal-learnedrouter-host-m0-source-v1"
DEF="${1:-MEAN_COMMITTEE}"
GPU="${2:-1}"
SEEDS="${SEEDS:-0,1,2}"
PY="/home/boyuanjie/envs/tif/bin/python"

"$PY" "$CODE/selftest.py"
"$PY" "$CODE/preflight.py"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$CODE/run_source_integration.py" \
  --stage1-definition "$DEF" \
  --seeds "$SEEDS" \
  --device auto
