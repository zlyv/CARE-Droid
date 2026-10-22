#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PCF_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${PCF_OUTPUT:-$PROJECT/outputs/care-p31-classifier-family-v1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PCF_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
mkdir -p "$OUTPUT/logs" "$OUTPUT/checkpoints"
exec > >(tee -a "$OUTPUT/logs/run.log") 2>&1
cd "$PROJECT"
phase() {
  local name="$1"; shift; local marker="$OUTPUT/checkpoints/${name}.done"
  if [[ -f "$marker" ]]; then echo "P31_FAMILY_PHASE_RESUME name=$name"; return; fi
  echo "P31_FAMILY_PHASE_START name=$name"; "$@"; touch "$marker"; echo "P31_FAMILY_PHASE_DONE name=$name"
}
echo "CARE_P31_CLASSIFIER_FAMILY_START output=$OUTPUT python=$PYTHON"
phase syntax "$PYTHON" -m py_compile "$HERE"/*.py
phase selftest "$PYTHON" "$HERE/selftest.py"
phase preflight "$PYTHON" "$HERE/preflight.py"
phase source_and_frozen "$PYTHON" "$HERE/evaluate_existing.py"
phase within_year "$PYTHON" "$HERE/run_within_year.py"
phase finalize "$PYTHON" "$HERE/finalize.py"
phase summary "$PYTHON" "$HERE/summarize.py"

