#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION="care-source-assets-v2"
LOG="artifacts/care-source-local-assets-v2.console.log"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" \
  "cd /home/boyuanjie/projects/tasa-droid && bash scripts/care-source-local-assets-v2/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -f $LOG"
echo "Attach: tmux attach -t $SESSION"
