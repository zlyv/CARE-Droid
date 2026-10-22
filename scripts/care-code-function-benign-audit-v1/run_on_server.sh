#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY="${CFBA_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
S=scripts/care-code-function-benign-audit-v1
O=outputs/care-code-function-benign-audit-v1
mkdir -p "$O/logs"
export CFBA_PROJECT=/home/boyuanjie/projects/tasa-droid
export CFBA_OUTPUT=/home/boyuanjie/projects/tasa-droid/outputs/care-code-function-benign-audit-v1
export PYTHONWARNINGS=ignore
echo "CARE_CODE_FUNCTION_BENIGN_AUDIT_START output=$O"
"$PY" -m py_compile "$S"/*.py
"$PY" "$S/selftest.py"
"$PY" "$S/preflight.py"
"$PY" "$S/run_audit.py"
"$PY" "$S/integrity_after.py"
echo "CARE_CODE_FUNCTION_BENIGN_AUDIT_DONE"
echo "decision=$O/final_decision.json"
