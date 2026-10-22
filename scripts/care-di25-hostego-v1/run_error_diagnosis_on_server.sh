#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/boyuanjie/projects/tasa-droid"
cd "$PROJECT"
PYTHON_BIN="${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}"
PACKAGE="scripts/care-di25-hostego-v1"
OUTPUT="artifacts/care-di25-hostego-v1/error_diagnosis"

export PYTHONNOUSERSITE=1
mkdir -p "$OUTPUT"

"$PYTHON_BIN" -c 'import numpy, sklearn; print("Dependencies:", numpy.__version__, sklearn.__version__)'
"$PYTHON_BIN" "$PACKAGE/selftest_error_diagnosis.py"

# Diagnostic-only: this calls no training entry point and writes only OUTPUT.
"$PYTHON_BIN" "$PACKAGE/run_sameyear_error_diagnosis.py" --output "$OUTPUT"

cat "$OUTPUT/FINAL_ERROR_DIAGNOSIS_REPORT.txt"

