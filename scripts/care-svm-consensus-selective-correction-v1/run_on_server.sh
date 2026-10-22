#!/usr/bin/env bash
set -euo pipefail

PROJECT="${SCSC_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${SCSC_OUTPUT:-$PROJECT/outputs/care-svm-consensus-selective-correction-v1}"
PYTHON="${SCSC_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTPUT/logs"
export SCSC_PROJECT="$PROJECT" SCSC_OUTPUT="$OUTPUT"

phase() {
  local name="$1"; shift
  echo "SCSC_PHASE_START name=$name epoch=$(date +%s)"
  "$@"
  echo "SCSC_PHASE_DONE name=$name epoch=$(date +%s)"
}

echo "CARE_SVM_CONSENSUS_SELECTIVE_CORRECTION_PIPELINE_START output=$OUTPUT"
phase syntax "$PYTHON" -m py_compile "$HERE"/*.py
phase selftest "$PYTHON" "$HERE/selftest.py"
phase preflight "$PYTHON" "$HERE/preflight.py"
phase source_audit "$PYTHON" "$HERE/run_source_audit.py"
phase summary "$PYTHON" "$HERE/summarize.py"
