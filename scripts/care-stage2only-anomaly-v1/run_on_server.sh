#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PYTHON_BIN="${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}"
OUTPUT="artifacts/care-stage2only-anomaly-v1/source"
PRED="$OUTPUT/oof_predictions.csv"
MALBEN="artifacts/care-stage2only-malben-v1/source/oof_predictions.csv"
mkdir -p "$OUTPUT"
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON_BIN" scripts/care-stage2only-anomaly-v1/preflight_stage2only_anomaly.py
"$PYTHON_BIN" scripts/care-stage2only-anomaly-v1/selftest_stage2only_anomaly.py
if "$PYTHON_BIN" scripts/care-stage2only-anomaly-v1/preflight_stage2only_anomaly.py \
  --check-output "$PRED"; then
  echo "Complete Stage-II-only Anomaly OOF already exists; skipping retraining."
else
  "$PYTHON_BIN" scripts/care-stage2only-anomaly-v1/run_stage2only_anomaly_v1.py \
    --seeds 0,1,2 --device auto \
    --cache artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz \
    --region-cache artifacts/care-source-local-assets-v2/regions_2017.npz \
    --output "$OUTPUT"
fi
"$PYTHON_BIN" scripts/care-stage2only-anomaly-v1/preflight_stage2only_anomaly.py \
  --check-output "$PRED"
"$PYTHON_BIN" scripts/care-stage2only-anomaly-v1/summarize_stage2only_comparison.py \
  --anomaly "$PRED" --malben "$MALBEN" --output "$OUTPUT"
cat "$OUTPUT/summary.txt"
cat "$OUTPUT/comparison.txt"
