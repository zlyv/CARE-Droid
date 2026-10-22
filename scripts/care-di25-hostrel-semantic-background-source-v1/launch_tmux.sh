#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION="care-hostrel-semantic-source-v1"
LOG="artifacts/care-di25-hostrel-semantic-background-source-v1.console.log"
GPU="${CARE_GPU:-0}"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "Already running: $SESSION"; exit 1; }
tmux new-session -d -s "$SESSION" "cd /home/boyuanjie/projects/tasa-droid && CARE_GPU='$GPU' bash scripts/care-di25-hostrel-semantic-background-source-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "GPU log: tail -F artifacts/care-di25-hostrel-semantic-background-source-v1/logs/R1_HOSTREL74_HARD_ROLE.gpu${GPU}.log"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
