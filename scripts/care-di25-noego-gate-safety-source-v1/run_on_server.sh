#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${CARE_ML_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "BLOCKED_BY_CONTRACT: missing ML Python $PYTHON_BIN" >&2; exit 2; }
"$PYTHON_BIN" scripts/care-di25-noego-gate-safety-source-v1/selftest.py
"$PYTHON_BIN" scripts/care-di25-noego-gate-safety-source-v1/preflight.py
"$PYTHON_BIN" scripts/care-di25-noego-gate-safety-source-v1/run_gate_safety_audit.py

