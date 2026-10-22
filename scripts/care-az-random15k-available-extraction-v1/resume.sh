#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
SCRIPT="$ROOT/scripts/care-az-random15k-available-extraction-v1"
OUT="$ROOT/outputs/care_az_random15k_available_extraction_v1"
PY=/home/boyuanjie/envs/tif/bin/python
export PYTHONDONTWRITEBYTECODE=1
test -f "$OUT/pending_future_lineage_label_free.csv"
test ! -f "$OUT/final_decision.json"
"$PY" "$SCRIPT/run_extraction.py"
"$PY" "$SCRIPT/finalize.py"

