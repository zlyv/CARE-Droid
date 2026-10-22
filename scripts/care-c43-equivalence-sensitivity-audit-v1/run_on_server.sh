#!/usr/bin/env bash
set -u

PROJECT="${CESA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${CESA_OUTPUT:-$PROJECT/outputs/care-c43-equivalence-sensitivity-audit-v2}"
PYTHON="${CESA_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
DEVICE="${CESA_DEVICE:-cuda}"
mkdir -p "$OUTPUT/logs" "$OUTPUT/checkpoints"
export CESA_PROJECT="$PROJECT" CESA_OUTPUT="$OUTPUT"

phase() {
  local name="$1"; shift
  local done_file="$OUTPUT/checkpoints/${name}.done"
  if [[ -f "$done_file" ]]; then echo "CESA_PHASE_RESUME name=$name"; return 0; fi
  echo "CESA_PHASE_START name=$name epoch=$(date +%s)"
  "$@" 2>&1 | tee "$OUTPUT/logs/${name}.log"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then echo "CESA_PHASE_FAIL name=$name rc=$rc"; exit "$rc"; fi
  touch "$done_file"; echo "CESA_PHASE_DONE name=$name epoch=$(date +%s)"
}

echo "CARE_C43_EQUIVALENCE_SENSITIVITY_START output=$OUTPUT python=$PYTHON device=$DEVICE"
phase syntax "$PYTHON" -m compileall -q "$SCRIPT_DIR"
phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
phase preflight "$PYTHON" "$SCRIPT_DIR/preflight.py"
phase audit "$PYTHON" "$SCRIPT_DIR/run_audit.py" --device "$DEVICE" --jobs "${CESA_JOBS:-4}" --repeats 3
phase summary "$PYTHON" "$SCRIPT_DIR/summarize.py"
phase integrity "$PYTHON" "$SCRIPT_DIR/integrity_after.py"
echo "CARE_C43_EQUIVALENCE_SENSITIVITY_PIPELINE_DONE"
echo "decision=$OUTPUT/final_decision.json"
echo "report=$OUTPUT/report.md"
