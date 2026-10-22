#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_EXE="${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}"
DIRECT_FILE="artifacts/care-rgcondattn-direct-l2-source-v1/source_predictions.csv"

export PYTHONNOUSERSITE=1

"${PYTHON_EXE}" scripts/care-di-directl2-comparison-v1/selftest_di_directl2_comparison.py

if [[ ! -f "${DIRECT_FILE}" ]]; then
  if [[ ! -f scripts/care-rgcondattn-direct-l2-source-v1/run_on_server.sh ]]; then
    echo "Missing ${DIRECT_FILE}"
    echo "Also missing scripts/care-rgcondattn-direct-l2-source-v1/run_on_server.sh"
    echo "Upload and unzip care-rgcondattn-direct-l2-source-v1.zip first, then rerun this script."
    exit 2
  fi
  echo "Direct-L2 OOF artifact is missing; running the established source experiment."
  PYTHON_BIN="${PYTHON_EXE}" \
    bash scripts/care-rgcondattn-direct-l2-source-v1/run_on_server.sh
fi

"${PYTHON_EXE}" scripts/care-di-directl2-comparison-v1/run_di_directl2_source_comparison_v1.py

echo "Source-only direct Stage-II 2x2 comparison completed."
