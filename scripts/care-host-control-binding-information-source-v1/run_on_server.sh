#!/usr/bin/env bash
set -euo pipefail
PROJECT="${HBI_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT="$PROJECT/scripts/care-host-control-binding-information-source-v1"
OUTPUT="$PROJECT/outputs/care-host-control-binding-information-source-v1"
if [ -e "$OUTPUT" ]; then echo "Output exists; use a new version, never overwrite."; exit 1; fi
export HBI_PROJECT="$PROJECT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT"
"${HBI_PYTHON:-/home/boyuanjie/envs/tif/bin/python}" -B "$SCRIPT/pipeline.py"
