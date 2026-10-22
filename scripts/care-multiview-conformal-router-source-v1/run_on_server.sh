#!/usr/bin/env bash
set -euo pipefail

cd /home/boyuanjie/projects/tasa-droid

PYTHON_BIN="${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}"
OUTPUT="${OUTPUT:-artifacts/care-multiview-conformal-router-source-v1}"

export PYTHONNOUSERSITE=1

"$PYTHON_BIN" \
  scripts/care-multiview-conformal-router-source-v1/selftest_multiview_conformal.py

"$PYTHON_BIN" \
  scripts/care-multiview-conformal-router-source-v1/run_multiview_conformal_router_source_v1.py \
  --expert-predictions \
  artifacts/care-rgcondattn-dedup-bounded-source-v1/source_predictions.csv \
  --budget 0.25 \
  --eta 1.0 \
  --inner-folds 4 \
  --calibration-fraction 0.25 \
  --platt-fraction 0.70 \
  --output "$OUTPUT"

cat "$OUTPUT/REPORT.txt"
