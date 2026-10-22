#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${CARE_ML_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "BLOCKED_BY_CONTRACT: missing ML Python: $PYTHON_BIN" >&2; exit 2
fi
"$PYTHON_BIN" scripts/care-di25-noego-decoupled-authority-source-v1/selftest.py
"$PYTHON_BIN" scripts/care-di25-noego-decoupled-authority-source-v1/preflight.py
"$PYTHON_BIN" scripts/care-di25-noego-decoupled-authority-source-v1/run_authority_audit.py

