#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PLRPD_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT="$PROJECT/scripts/care-p31-lr-pairwise-diagnostics-v1"
OUTPUT="${PLRPD_OUTPUT:-$PROJECT/outputs/care-p31-lr-pairwise-diagnostics-v1}"
PY="${PLRPD_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
if [[ -e "$OUTPUT" ]]; then echo "BLOCKED_BY_CONTRACT: output exists: $OUTPUT" >&2; exit 1; fi
mkdir -p "$OUTPUT/logs"
exec > >(tee -a "$OUTPUT/logs/run.log") 2>&1
echo "CARE_P31_LR_PAIRWISE_DIAGNOSTICS_START output=$OUTPUT"
"$PY" -m py_compile "$SCRIPT"/*.py
PYTHONPATH="$SCRIPT" "$PY" "$SCRIPT/selftest.py"
PYTHONPATH="$SCRIPT" PLRPD_PROJECT="$PROJECT" PLRPD_OUTPUT="$OUTPUT" "$PY" "$SCRIPT/run_diagnostics.py"

