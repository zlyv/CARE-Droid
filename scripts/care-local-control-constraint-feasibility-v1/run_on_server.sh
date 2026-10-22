#!/usr/bin/env bash
set -euo pipefail
PROJECT="${LCC_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT="$PROJECT/scripts/care-local-control-constraint-feasibility-v1"
OUTPUT="${LCC_OUTPUT:-$PROJECT/outputs/care-local-control-constraint-feasibility-v1}"
PYTHON="${LCC_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
mkdir -p "$OUTPUT/logs"
export LCC_PROJECT="$PROJECT" LCC_OUTPUT="$OUTPUT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$PROJECT"
"$PYTHON" "$SCRIPT/pipeline.py"
