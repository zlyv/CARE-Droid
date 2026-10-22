#!/usr/bin/env bash
set -euo pipefail

cd /home/boyuanjie/projects/tasa-droid

PYTHON_BIN="${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}"
OUTPUT="${OUTPUT:-artifacts/care-malcertain-router-source-v1}"

export PYTHONNOUSERSITE=1

"$PYTHON_BIN" scripts/care-malcertain-router-source-v1/selftest_malcertain_router.py
"$PYTHON_BIN" scripts/care-malcertain-router-source-v1/run_care_malcertain_router_source_v1.py \
  --expert-predictions \
  artifacts/care-rgcondattn-dedup-bounded-source-v1/source_predictions.csv \
  --budget 0.25 \
  --inner-folds 4 \
  --output "$OUTPUT" \
  "$@"

cat "$OUTPUT/REPORT.txt"
