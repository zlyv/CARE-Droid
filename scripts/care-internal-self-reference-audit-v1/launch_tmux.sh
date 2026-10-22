#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
SESSION=care-internal-self-reference-audit-v1
LOG="$ROOT/outputs/care_internal_self_reference_audit_v1.console.log"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION" >&2
  exit 1
fi
if [ -e "$ROOT/outputs/care_internal_self_reference_audit_v1" ] || [ -e "$LOG" ]; then
  echo "Output already exists; refusing overwrite" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" "cd '$ROOT' && ISR_DEVICE='${ISR_DEVICE:-cpu}' bash scripts/care-internal-self-reference-audit-v1/run.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"

