#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PLRF_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SESSION="care-p31-lr-source-safe-fusion-v1"
LOG="$PROJECT/outputs/care-p31-lr-source-safe-fusion-v1.console.log"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION" >&2; exit 1
fi
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && bash scripts/care-p31-lr-source-safe-fusion-v1/run_on_server.sh > '$LOG' 2>&1"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Attach: tmux attach -t $SESSION"

