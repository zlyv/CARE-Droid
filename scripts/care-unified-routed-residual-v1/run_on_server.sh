#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY="${CURR_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
S=scripts/care-unified-routed-residual-v1
O=outputs/care-unified-routed-residual-v1
mkdir -p "$O/logs"
export CURR_PROJECT=/home/boyuanjie/projects/tasa-droid
export CURR_OUTPUT=/home/boyuanjie/projects/tasa-droid/outputs/care-unified-routed-residual-v1
export PYTHONWARNINGS=ignore
echo "CARE_UNIFIED_ROUTED_RESIDUAL_START output=$O python=$PY"
"$PY" -m py_compile "$S"/*.py
"$PY" "$S/selftest.py"
"$PY" "$S/preflight.py"
"$PY" "$S/run_probe.py"
"$PY" "$S/integrity_after.py"
echo "CARE_UNIFIED_ROUTED_RESIDUAL_DONE"
echo "decision=$O/final_decision.json"
echo "table=$O/routed_200_results.csv"
