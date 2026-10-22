#!/usr/bin/env bash
set -uo pipefail
PROJECT="${JCIA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${JCIA_OUTPUT:-$PROJECT/outputs/care-direct-c43-joint-contribution-audit-v1}"
PYTHON="${JCIA_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
export JCIA_PROJECT="$PROJECT" JCIA_OUTPUT="$OUTPUT" PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
mkdir -p "$OUTPUT/logs" "$OUTPUT/audits"
phase(){
  local name="$1"; shift
  local done="$OUTPUT/checkpoint_${name}.done"
  if [[ -f "$done" ]]; then echo "JCIA_PHASE_RESUME name=$name"; return 0; fi
  echo "JCIA_PHASE_START name=$name epoch=$(date +%s)"
  "$@" 2>&1 | tee "$OUTPUT/logs/${name}.log"; local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then echo "JCIA_PHASE_FAIL name=$name rc=$rc"; exit "$rc"; fi
  touch "$done"; echo "JCIA_PHASE_DONE name=$name epoch=$(date +%s)"
}
echo "CARE_DIRECT_C43_JOINT_AUDIT_START output=$OUTPUT python=$PYTHON"
phase syntax "$PYTHON" -m py_compile "$SCRIPT_DIR"/*.py
phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
phase preflight "$PYTHON" "$SCRIPT_DIR/preflight.py"
phase audit "$PYTHON" "$SCRIPT_DIR/run_audit.py" --device "${JCIA_DEVICE:-cuda}" --jobs "${JCIA_JOBS:-4}" --repeats 12
phase summary "$PYTHON" "$SCRIPT_DIR/summarize.py"
phase integrity "$PYTHON" "$SCRIPT_DIR/integrity_after.py"
echo "CARE_DIRECT_C43_JOINT_AUDIT_DONE"
echo "decision=$OUTPUT/final_decision.json"
echo "report=$OUTPUT/REPORT.md"

