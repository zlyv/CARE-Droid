#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x /home/boyuanjie/envs/tif/bin/python ]]; then
  PYTHON_BIN=/home/boyuanjie/envs/tif/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "No usable Python interpreter was found." >&2
  exit 127
fi

export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON_BIN" \
  scripts/care-reliability-calibrated-evidential-router-v2/selftest_reliability_calibrated.py

DIRECT_PREDICTIONS=artifacts/care-rgcondattn-direct-l2-source-v1/source_predictions.csv
if [[ ! -f "$DIRECT_PREDICTIONS" ]]; then
  echo "Direct-L2 OOF predictions not found; running the source ablation first."
  bash scripts/care-rgcondattn-direct-l2-source-v1/run_on_server.sh
fi

"$PYTHON_BIN" \
  scripts/care-reliability-calibrated-evidential-router-v2/run_reliability_calibrated_source_v2.py \
  --device auto

"$PYTHON_BIN" \
  scripts/care-reliability-calibrated-evidential-router-v2/run_reliability_calibrated_future_v2.py \
  --device auto

cat artifacts/care-reliability-calibrated-evidential-router-source-v2/REPORT.txt
cat artifacts/care-reliability-calibrated-evidential-router-future-v2/REPORT.txt

