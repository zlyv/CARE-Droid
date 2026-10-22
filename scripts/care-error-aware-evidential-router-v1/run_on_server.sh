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

"$PYTHON_BIN" scripts/care-error-aware-evidential-router-v1/selftest_error_aware_evidential.py

"$PYTHON_BIN" scripts/care-error-aware-evidential-router-v1/run_error_aware_evidential_source_v1.py \
  --device auto

"$PYTHON_BIN" scripts/care-error-aware-evidential-router-v1/run_error_aware_evidential_future_v1.py \
  --device auto

cat artifacts/care-error-aware-evidential-router-source-v1/REPORT.txt
cat artifacts/care-error-aware-evidential-router-future-v1/REPORT.txt

