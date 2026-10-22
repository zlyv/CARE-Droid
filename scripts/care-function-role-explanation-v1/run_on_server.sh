#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY="${CFRE_PYTHON:-/home/boyuanjie/envs/tif/bin/python}";S=scripts/care-function-role-explanation-v1;O=outputs/care-function-role-explanation-v1
mkdir -p "$O/logs";export CFRE_PROJECT=/home/boyuanjie/projects/tasa-droid;export CFRE_OUTPUT=/home/boyuanjie/projects/tasa-droid/outputs/care-function-role-explanation-v1;export LOGURU_LEVEL=ERROR;export PYTHONWARNINGS=ignore
echo "CARE_FUNCTION_ROLE_EXPLANATION_START static_only=true output=$O"
"$PY" -m py_compile "$S"/*.py;"$PY" "$S/selftest.py";"$PY" "$S/preflight.py";"$PY" "$S/extract_features.py" --jobs "${CFRE_JOBS:-4}";"$PY" "$S/run_probe.py";"$PY" "$S/integrity_after.py"
echo "CARE_FUNCTION_ROLE_EXPLANATION_DONE";echo "decision=$O/final_decision.json";echo "probe=$O/probe_results.json"
