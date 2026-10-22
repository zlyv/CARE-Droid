#!/usr/bin/env bash
set -euo pipefail
PROJECT="${SCSCD_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${SCSCD_OUTPUT:-$PROJECT/outputs/care-svm-consensus-selective-correction-diagnostics-v1}"
PYTHON="${SCSCD_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTPUT/logs";export SCSCD_PROJECT="$PROJECT" SCSCD_OUTPUT="$OUTPUT"
phase(){ local n="$1";shift;echo "SCSCD_PHASE_START name=$n epoch=$(date +%s)";"$@";echo "SCSCD_PHASE_DONE name=$n epoch=$(date +%s)";}
echo "CARE_SVM_CONSENSUS_DIAGNOSTICS_PIPELINE_START output=$OUTPUT"
phase syntax "$PYTHON" -m py_compile "$HERE"/*.py
phase selftest "$PYTHON" "$HERE/selftest.py"
phase preflight "$PYTHON" "$HERE/preflight.py"
phase diagnostics "$PYTHON" "$HERE/run_diagnostics.py"
phase summary "$PYTHON" "$HERE/summarize.py"
