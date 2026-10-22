#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/boyuanjie/projects/tasa-droid
SCRIPT="$PROJECT/scripts/care-group-calibration-compact43-validation-v1"
OUTPUT="$PROJECT/outputs/care-group-calibration-compact43-validation-v1"
PYTHON=${CARE_PYTHON:-/home/boyuanjie/envs/tif/bin/python}

cd "$PROJECT"
if [[ ! -x "$PYTHON" ]]; then
  echo "BLOCKED_BY_CONTRACT: Python not executable: $PYTHON"
  exit 1
fi
if [[ -e "$OUTPUT" ]]; then
  echo "BLOCKED_BY_CONTRACT: output exists; refusing to overwrite: $OUTPUT"
  exit 1
fi
"$PYTHON" -m py_compile "$SCRIPT"/*.py
"$PYTHON" "$SCRIPT/selftest.py"
"$PYTHON" "$SCRIPT/integration_selftest.py"
"$PYTHON" "$SCRIPT/preflight.py"
"$PYTHON" "$SCRIPT/run_validation.py"
"$PYTHON" "$SCRIPT/summarize.py"

