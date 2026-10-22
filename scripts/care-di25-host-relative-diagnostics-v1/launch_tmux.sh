#!/usr/bin/env bash
set -euo pipefail
PROJECT="/home/boyuanjie/projects/tasa-droid"
SESSION="care-host-relative-diagnostics-v1"
LOG="$PROJECT/artifacts/care-di25-host-relative-diagnostics-v1.console.log"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi
mkdir -p "$PROJECT/artifacts"
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && bash scripts/care-di25-host-relative-diagnostics-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "GPU job logs: $PROJECT/artifacts/care-di25-host-relative-diagnostics-v1/logs/"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
