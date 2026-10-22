#!/usr/bin/env bash
set -euo pipefail

PROJECT="${CSCUA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${CSCUA_OUTPUT:-$PROJECT/outputs/care-svm-c43-unfixed-audit-v1}"
PYTHON="${CSCUA_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"

case "$OUTPUT" in
  "$PROJECT"/outputs/care-svm-c43-unfixed-audit-v1) ;;
  *) echo "BLOCKED_BY_CONTRACT: unsafe output path: $OUTPUT"; exit 1 ;;
esac
test -x "$PYTHON" || { echo "BLOCKED_BY_CONTRACT: Python missing: $PYTHON"; exit 1; }
mkdir -p "$OUTPUT/logs"
export CSCUA_PROJECT="$PROJECT" CSCUA_OUTPUT="$OUTPUT" PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT"

phase() {
  local name="$1"; shift
  echo "CSCUA_PHASE_START name=$name epoch=$(date +%s)"
  "$@"
  echo "CSCUA_PHASE_DONE name=$name epoch=$(date +%s)"
}

echo "CARE_SVM_C43_UNFIXED_AUDIT_START output=$OUTPUT python=$PYTHON"
phase syntax "$PYTHON" -m compileall -q "$SCRIPT_DIR"
phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
phase preflight "$PYTHON" "$SCRIPT_DIR/preflight.py"
phase audit "$PYTHON" "$SCRIPT_DIR/run_audit.py"
phase integrity "$PYTHON" "$SCRIPT_DIR/integrity_after.py"
phase summary "$PYTHON" "$SCRIPT_DIR/summarize.py"
echo "report=$OUTPUT/REPORT.md"
echo "decision=$OUTPUT/final_decision.json"
