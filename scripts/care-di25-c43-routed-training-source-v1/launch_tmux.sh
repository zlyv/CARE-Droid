#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION="care-c43-routed-training-v1"
LOG="artifacts/care-di25-c43-routed-training-source-v1.console.log"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Already running: $SESSION"; exit 0; fi
tmux new-session -d -s "$SESSION" "cd /home/boyuanjie/projects/tasa-droid && bash scripts/care-di25-c43-routed-training-source-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION";echo "Monitor: tail -F $LOG";echo "GPU logs: artifacts/care-di25-c43-routed-training-source-v1/logs/";echo "Attach: tmux attach -t $SESSION";echo "Detach: Ctrl-b, then d"

