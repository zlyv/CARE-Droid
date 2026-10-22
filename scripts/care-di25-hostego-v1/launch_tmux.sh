#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PHASE="${1:-m01}"
SESSION="care-di25-hostego-${PHASE}"
LOG="artifacts/care-di25-hostego-v1.${PHASE}.console.log"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"; exit 1
fi
tmux new-session -d -s "$SESSION" "cd /home/boyuanjie/projects/tasa-droid && bash scripts/care-di25-hostego-v1/run_on_server.sh '$PHASE' 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Attach: tmux attach -t $SESSION"

