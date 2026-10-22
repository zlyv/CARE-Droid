#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION="care-compact43-relationquery-v1";LOG="artifacts/care-di25-compact43-relationquery-source-v1.console.log"
tmux has-session -t "$SESSION" 2>/dev/null&&{ echo "Already running: $SESSION";exit 0; }
tmux new-session -d -s "$SESSION" "cd /home/boyuanjie/projects/tasa-droid && bash scripts/care-di25-compact43-relationquery-source-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION";echo "Monitor: tail -F $LOG";echo "GPU logs: artifacts/care-di25-compact43-relationquery-source-v1/logs/";echo "Attach: tmux attach -t $SESSION";echo "Detach: Ctrl-b, then d"
