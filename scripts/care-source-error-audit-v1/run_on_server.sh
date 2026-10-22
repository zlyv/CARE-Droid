#!/usr/bin/env bash
set -euo pipefail

cd /home/boyuanjie/projects/tasa-droid
PYTHON_BIN="${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}"
export PYTHONNOUSERSITE=1

"$PYTHON_BIN" \
  scripts/care-source-error-audit-v1/analyze_care_source_errors.py \
  --predictions artifacts/care-asymmetric-a25-router-source-v1/source_predictions.csv \
  --features artifacts/p1-features/features.csv \
  --output artifacts/care-source-error-audit-v1
