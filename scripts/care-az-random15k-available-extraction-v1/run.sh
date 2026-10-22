#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
SCRIPT="$ROOT/scripts/care-az-random15k-available-extraction-v1"
OUT="$ROOT/outputs/care_az_random15k_available_extraction_v1"
PY=/home/boyuanjie/envs/tif/bin/python
export PYTHONDONTWRITEBYTECODE=1
echo "CARE_AZ_AVAILABLE_EXTRACTION_START output=$OUT"
"$PY" -m py_compile "$SCRIPT"/*.py
"$PY" "$SCRIPT/selftest.py"
"$PY" "$SCRIPT/prepare_snapshot.py"
"$PY" "$SCRIPT/run_extraction.py"
"$PY" "$SCRIPT/finalize.py"

