#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION="care-litesemantic-sizecond-v1";LOG="artifacts/care-litesemantic-sizeconditioned-source-v1.console.log"
if tmux has-session -t "$SESSION" 2>/dev/null;then echo "Already running: $SESSION";exit 0;fi
tmux new-session -d -s "$SESSION" "cd /home/boyuanjie/projects/tasa-droid && bash scripts/care-litesemantic-sizeconditioned-source-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION";echo "Monitor: tail -F $LOG";echo "Attach: tmux attach -t $SESSION"
