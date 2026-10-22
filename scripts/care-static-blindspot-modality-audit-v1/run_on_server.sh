#!/usr/bin/env bash
set -uo pipefail
PROJECT="${SBMA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${SBMA_OUTPUT:-$PROJECT/outputs/care-static-blindspot-modality-audit-v1}"
PYTHON="${SBMA_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
mkdir -p "$OUTPUT/logs" "$OUTPUT/checkpoints" "$OUTPUT/cache"
LOG="$OUTPUT/logs/run.log"
phase() {
  local name="$1";shift;local marker="$OUTPUT/checkpoints/${name}.done"
  if [[ -f "$marker" ]];then echo "SBMA_PHASE_RESUME name=$name" | tee -a "$LOG";return 0;fi
  echo "SBMA_PHASE_START name=$name epoch=$(date +%s)" | tee -a "$LOG"
  if "$@" 2>&1 | tee -a "$LOG";then printf 'done\n' > "$marker";echo "SBMA_PHASE_DONE name=$name epoch=$(date +%s)" | tee -a "$LOG"
  else local rc=${PIPESTATUS[0]};echo "SBMA_PHASE_FAIL name=$name rc=$rc epoch=$(date +%s)" | tee -a "$LOG";exit "$rc";fi
}
export SBMA_PROJECT="$PROJECT" SBMA_OUTPUT="$OUTPUT"
echo "CARE_STATIC_BLINDSPOT_MODALITY_AUDIT_START output=$OUTPUT python=$PYTHON workers=${SBMA_WORKERS:-3}" | tee -a "$LOG"
phase syntax "$PYTHON" -m py_compile "$SCRIPT_DIR"/*.py
phase selftest "$PYTHON" "$SCRIPT_DIR/selftest.py"
phase preflight "$PYTHON" "$SCRIPT_DIR/preflight.py"
phase audit "$PYTHON" "$SCRIPT_DIR/run_audit.py"
echo "CARE_STATIC_BLINDSPOT_MODALITY_AUDIT_DONE" | tee -a "$LOG"
echo "decision=$OUTPUT/final_decision.json" | tee -a "$LOG"
echo "report=$OUTPUT/REPORT.md" | tee -a "$LOG"
