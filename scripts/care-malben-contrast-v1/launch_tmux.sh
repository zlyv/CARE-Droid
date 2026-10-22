#!/usr/bin/env bash
set -euo pipefail

cd /home/boyuanjie/projects/tasa-droid
mkdir -p artifacts

SESSION="${SESSION:-care-malben-v1}"
LOG="artifacts/care-malben-contrast-v1.console.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd /home/boyuanjie/projects/tasa-droid && bash scripts/care-malben-contrast-v1/run_on_server.sh 2>&1 | tee '$LOG'"

echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Log: tail -f $LOG"

