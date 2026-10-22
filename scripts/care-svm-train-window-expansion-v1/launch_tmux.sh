#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION="care-svm-train-window-v1"
LOG="/home/boyuanjie/projects/tasa-droid/outputs/care-svm-train-window-expansion-v1.console.log"
mkdir -p outputs
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Already running: $SESSION"; exit 0
fi
tmux new-session -d -s "$SESSION" "bash scripts/care-svm-train-window-expansion-v1/run_on_server.sh 2>&1 | tee -a '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
