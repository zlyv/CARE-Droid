#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION="care-malben-host-v2"
LOG="artifacts/care-malben-host-v2.console.log"
PRECHECKED="${CARE_GATES_PRECHECKED:-0}"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session exists: $SESSION" >&2; exit 1; }
tmux new-session -d -s "$SESSION" \
  "cd /home/boyuanjie/projects/tasa-droid && CARE_GATES_PRECHECKED='$PRECHECKED' bash scripts/care-malben-host-v2/run_selective_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -f $LOG"
