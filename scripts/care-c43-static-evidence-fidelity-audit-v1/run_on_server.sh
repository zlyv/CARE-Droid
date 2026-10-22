#!/usr/bin/env bash
set -euo pipefail

PROJECT="${CSEF_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT="$PROJECT/scripts/care-c43-static-evidence-fidelity-audit-v1"
OUTPUT="${CSEF_OUTPUT:-$PROJECT/outputs/care-c43-static-evidence-fidelity-audit-v1}"
PYTHON="${CSEF_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
mkdir -p "$OUTPUT/logs"
export CSEF_PROJECT="$PROJECT" CSEF_OUTPUT="$OUTPUT" PYTHONPATH="$SCRIPT${PYTHONPATH:+:$PYTHONPATH}"

phase() {
  local name="$1" file="$2"
  echo "CSEF_PHASE_START name=$name epoch=$(date +%s)"
  "$PYTHON" "$SCRIPT/$file"
  echo "CSEF_PHASE_DONE name=$name epoch=$(date +%s)"
}

echo "CARE_C43_STATIC_EVIDENCE_FIDELITY_START output=$OUTPUT workers=${CSEF_WORKERS:-4}"
echo "CSEF_PHASE_START name=syntax epoch=$(date +%s)"
"$PYTHON" -m py_compile "$SCRIPT"/*.py
echo "CSEF_PHASE_DONE name=syntax epoch=$(date +%s)"
phase selftest selftest.py
phase preflight preflight.py
phase audit run_audit.py
phase review_summary summarize_review.py
phase integrity integrity.py
phase report report.py
echo "CARE_C43_STATIC_EVIDENCE_FIDELITY_DONE"
echo "decision=$OUTPUT/final_decision.json"
echo "report=$OUTPUT/REPORT.md"
