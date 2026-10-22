#!/usr/bin/env bash
set -uo pipefail
PROJECT="${CRC_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${CRC_OUTPUT:-$PROJECT/outputs/care-svm-c43-representation-collision-audit-v1}"
PYTHON="${CRC_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
export CRC_PROJECT="$PROJECT" CRC_OUTPUT="$OUTPUT" PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
mkdir -p "$OUTPUT/logs" "$OUTPUT/audits"
phase(){
  local name="$1"; shift
  local done="$OUTPUT/checkpoint_${name}.done"
  if [[ -f "$done" ]]; then echo "CRC_PHASE_RESUME name=$name"; return 0; fi
  echo "CRC_PHASE_START name=$name epoch=$(date +%s)"
  "$@" 2>&1 | tee "$OUTPUT/logs/${name}.log"; local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then echo "CRC_PHASE_FAIL name=$name rc=$rc"; exit "$rc"; fi
  touch "$done"; echo "CRC_PHASE_DONE name=$name epoch=$(date +%s)"
}
echo "CARE_C43_REPRESENTATION_COLLISION_PIPELINE_START output=$OUTPUT python=$PYTHON"
phase syntax "$PYTHON" -m py_compile "$SCRIPT_DIR"/*.py
phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
phase preflight "$PYTHON" "$SCRIPT_DIR/preflight.py"
phase replay "$PYTHON" "$SCRIPT_DIR/run_replay.py" --device "${CRC_DEVICE:-cuda}" --jobs "${CRC_JOBS:-4}"
phase collision_audit "$PYTHON" "$SCRIPT_DIR/audit_collision.py"
phase summary "$PYTHON" "$SCRIPT_DIR/summarize.py"
phase integrity "$PYTHON" "$SCRIPT_DIR/integrity_after.py"
echo "CARE_C43_REPRESENTATION_COLLISION_PIPELINE_DONE"
echo "decision=$OUTPUT/final_decision.json"
echo "report=$OUTPUT/REPORT.md"
