#!/usr/bin/env bash
set -uo pipefail

PROJECT="${CELA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${CELA_OUTPUT:-$PROJECT/outputs/care-consensus-error-lineage-label-audit-v1}"
PYTHON="${CELA_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
mkdir -p "$OUTPUT/logs" "$OUTPUT/checkpoints" "$OUTPUT/cache"
LOG="$OUTPUT/logs/run.log"

phase() {
  local name="$1"; shift
  local marker="$OUTPUT/checkpoints/${name}.done"
  if [[ -f "$marker" ]]; then
    echo "CELA_PHASE_RESUME name=$name" | tee -a "$LOG"
    return 0
  fi
  echo "CELA_PHASE_START name=$name epoch=$(date +%s)" | tee -a "$LOG"
  if "$@" 2>&1 | tee -a "$LOG"; then
    printf 'done\n' > "$marker"
    echo "CELA_PHASE_DONE name=$name epoch=$(date +%s)" | tee -a "$LOG"
  else
    local rc=${PIPESTATUS[0]}
    echo "CELA_PHASE_FAIL name=$name rc=$rc epoch=$(date +%s)" | tee -a "$LOG"
    exit "$rc"
  fi
}

export CELA_PROJECT="$PROJECT" CELA_OUTPUT="$OUTPUT"
echo "CARE_CONSENSUS_ERROR_LINEAGE_LABEL_AUDIT_START output=$OUTPUT python=$PYTHON workers=${CELA_WORKERS:-4}" | tee -a "$LOG"
phase syntax "$PYTHON" -m py_compile "$SCRIPT_DIR"/*.py
phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
phase preflight "$PYTHON" "$SCRIPT_DIR/preflight.py"
phase audit "$PYTHON" "$SCRIPT_DIR/run_audit.py"
echo "CARE_CONSENSUS_ERROR_LINEAGE_LABEL_AUDIT_DONE" | tee -a "$LOG"
echo "decision=$OUTPUT/final_decision.json" | tee -a "$LOG"
echo "report=$OUTPUT/REPORT.md" | tee -a "$LOG"
