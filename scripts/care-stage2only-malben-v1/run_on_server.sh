#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PYTHON_BIN="${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}"
OUTPUT="artifacts/care-stage2only-malben-v1/source"
PRED="$OUTPUT/oof_predictions.csv"
CHECK="scripts/care-source-recovery-stage2only-v1/check_complete.py"
mkdir -p "$OUTPUT"
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "${CARE_GATES_PRECHECKED:-0}" != "1" ]]; then
  bash scripts/care-malben-host-v2/run_all_gates_on_server.sh
fi
if "$PYTHON_BIN" "$CHECK" csv "$PRED" --expected 799; then
  echo "Complete Stage-II-only OOF already exists; skipping retraining."
else
  "$PYTHON_BIN" scripts/care-stage2only-malben-v1/run_stage2only_malben_v1.py \
    --seeds 0,1,2 --device auto \
    --cache artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz \
    --region-cache artifacts/care-source-local-assets-v2/regions_2017.npz \
    --output "$OUTPUT"
fi
"$PYTHON_BIN" "$CHECK" csv "$PRED" --expected 799
cat "$OUTPUT/summary.txt"
