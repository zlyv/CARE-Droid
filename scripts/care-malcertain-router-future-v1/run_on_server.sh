#!/usr/bin/env bash
set -euo pipefail

cd /home/boyuanjie/projects/tasa-droid

PYTHON_BIN="${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}"
OUTPUT="${OUTPUT:-artifacts/care-malcertain-router-future-v1}"
FUTURE_EXPERT="${FUTURE_EXPERT:-artifacts/care-rgcondattn-dedup-bounded-future-diagnostic-v1/future_predictions_diagnostic.csv}"

export PYTHONNOUSERSITE=1

if [[ ! -f "$FUTURE_EXPERT" ]]; then
  echo "Missing frozen future expert predictions: $FUTURE_EXPERT" >&2
  echo "Run care-rgcondattn-dedup-bounded-future-v1 first." >&2
  exit 2
fi

"$PYTHON_BIN" scripts/care-malcertain-router-source-v1/selftest_malcertain_router.py
"$PYTHON_BIN" scripts/care-malcertain-router-future-v1/selftest_future_router.py
"$PYTHON_BIN" scripts/care-malcertain-router-future-v1/run_care_malcertain_router_future_v1.py \
  --future-expert-predictions "$FUTURE_EXPERT" \
  --years 2018,2019,2020,2021 \
  --budget 0.25 \
  --inner-folds 4 \
  --seed 20260914 \
  --output "$OUTPUT" \
  "$@"

cat "$OUTPUT/REPORT.txt"

