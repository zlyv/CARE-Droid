#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PYTHON_BIN="${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}"
CHECK="scripts/care-source-recovery-stage2only-v1/check_complete.py"
ROOT="artifacts/care-malben-host-v2/source"
ANOMALY="$ROOT/anomaly/source_predictions.csv"
MALBEN="$ROOT/malben/source_predictions.csv"
mkdir -p "$ROOT"
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "${CARE_GATES_PRECHECKED:-0}" != "1" ]]; then
  bash scripts/care-malben-host-v2/run_all_gates_on_server.sh
fi

if "$PYTHON_BIN" "$CHECK" csv "$ANOMALY" --expected 799; then
  echo "Complete anomaly OOF already exists; skipping retraining."
else
  "$PYTHON_BIN" scripts/care-malben-host-v2/run_anomaly_baseline_v2.py \
    --variants RGCondAttn --seeds 0,1,2 --max-delta 2.25 --device auto \
    --cache artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz \
    --region-cache artifacts/care-source-local-assets-v2/regions_2017.npz \
    --output "$ROOT/anomaly"
fi
"$PYTHON_BIN" "$CHECK" csv "$ANOMALY" --expected 799

if "$PYTHON_BIN" "$CHECK" csv "$MALBEN" --expected 799; then
  echo "Complete MalBen OOF already exists; skipping retraining."
else
  "$PYTHON_BIN" scripts/care-malben-host-v2/run_malben_host_v2.py \
    --variants MalBenContrast --seeds 0,1,2 --max-delta 2.25 --device auto \
    --cache artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz \
    --region-cache artifacts/care-source-local-assets-v2/regions_2017.npz \
    --output "$ROOT/malben"
fi
"$PYTHON_BIN" "$CHECK" csv "$MALBEN" --expected 799

"$PYTHON_BIN" scripts/care-malben-host-v2/analyze_source_v2.py \
  --anomaly "$ANOMALY" --malben "$MALBEN" --output "$ROOT"
cat "$ROOT/summary.txt"
