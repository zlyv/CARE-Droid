#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-p31rf-dependency-probe-v1
LOG=artifacts/care-p31rf-c43-dependency-sufficiency-source-v1.console.log
tmux has-session -t "$SESSION" 2>/dev/null && { echo "Already running: $SESSION"; exit 0; }
tmux new-session -d -s "$SESSION" "cd /home/boyuanjie/projects/tasa-droid && bash scripts/care-p31rf-c43-dependency-sufficiency-source-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Attach: tmux attach -t $SESSION"

