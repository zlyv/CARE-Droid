#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PLRF_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT="$PROJECT/scripts/care-p31-lr-source-safe-fusion-v1"
OUTPUT="${PLRF_OUTPUT:-$PROJECT/outputs/care-p31-lr-source-safe-fusion-v1}"
PY="${PLRF_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"

if [[ -e "$OUTPUT" ]]; then
  echo "BLOCKED_BY_CONTRACT: output already exists: $OUTPUT" >&2
  exit 1
fi
mkdir -p "$OUTPUT/logs"
exec > >(tee -a "$OUTPUT/logs/run.log") 2>&1

echo "CARE_P31_LR_SOURCE_SAFE_FUSION_START output=$OUTPUT python=$PY"
"$PY" -m py_compile "$SCRIPT"/*.py
PYTHONPATH="$SCRIPT" "$PY" "$SCRIPT/selftest.py"
PYTHONPATH="$SCRIPT" PLRF_PROJECT="$PROJECT" PLRF_OUTPUT="$OUTPUT" "$PY" "$SCRIPT/run_source.py"
echo "CARE_P31_LR_SOURCE_SAFE_FUSION_DONE"
echo "decision=$OUTPUT/final_decision.json"
echo "metrics=$OUTPUT/source_metrics.csv"

