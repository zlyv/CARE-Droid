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

"$PYTHON_BIN" scripts/care-evidential-conflict-router-v1/selftest_evidential_conflict.py

"$PYTHON_BIN" scripts/care-evidential-conflict-router-v1/run_evidential_conflict_source_v1.py \
  --device auto

"$PYTHON_BIN" scripts/care-evidential-conflict-router-v1/run_evidential_conflict_future_v1.py \
  --device auto

cat artifacts/care-evidential-conflict-router-source-v1/REPORT.txt
cat artifacts/care-evidential-conflict-router-future-v1/REPORT.txt

