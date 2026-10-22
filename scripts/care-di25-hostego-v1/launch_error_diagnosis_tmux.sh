#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/boyuanjie/projects/tasa-droid"
SESSION="care-error-diagnosis-v1"
LOG="$PROJECT/artifacts/care-di25-hostego-v1/error_diagnosis.console.log"
cd "$PROJECT"
mkdir -p "$(dirname "$LOG")"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "cd '$PROJECT' && bash scripts/care-di25-hostego-v1/run_error_diagnosis_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Attach: tmux attach -t $SESSION"

