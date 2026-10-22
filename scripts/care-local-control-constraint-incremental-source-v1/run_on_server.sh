#!/usr/bin/env bash
set -euo pipefail
PROJECT="${LCIP_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${LCIP_OUTPUT:-$PROJECT/outputs/care-local-control-constraint-incremental-source-v1}"
if [ "$OUTPUT" != "$PROJECT/outputs/care-local-control-constraint-incremental-source-v1" ]; then
  echo "Refusing non-experiment output directory: $OUTPUT"; exit 1
fi
export LCIP_PROJECT="$PROJECT" LCIP_OUTPUT="$OUTPUT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT"
"${LCIP_PYTHON:-/home/boyuanjie/envs/tif/bin/python}" -B "$PROJECT/scripts/care-local-control-constraint-incremental-source-v1/pipeline.py"
