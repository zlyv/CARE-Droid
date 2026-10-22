#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PYTHON_BIN="${CARE_ML_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
SCRIPT="scripts/care-di25-hostego-v1"

"$PYTHON_BIN" "$SCRIPT/selftest_final_validation.py"
"$PYTHON_BIN" "$SCRIPT/freeze_final_method.py"
"$PYTHON_BIN" "$SCRIPT/audit_final_protocol.py" --bootstrap 10000
"$PYTHON_BIN" "$SCRIPT/run_seed_robustness.py" --mode dry-run --replicate R1

echo FINAL_VALIDATION_PREFLIGHT_PASS
echo "Steps 1-6 passed. Inspect the contracts, then launch seed robustness explicitly."
