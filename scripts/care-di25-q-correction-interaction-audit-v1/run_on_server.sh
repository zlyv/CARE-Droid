#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${CARE_ML_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
SCRIPT="scripts/care-di25-q-correction-interaction-audit-v1"
"$PY" "$SCRIPT/selftest.py"
"$PY" "$SCRIPT/preflight.py"
"$PY" "$SCRIPT/run_audit.py"
