#!/usr/bin/env bash
set -euo pipefail
PROJECT="/home/boyuanjie/projects/tasa-droid";SESSION="care-roleconditioned-bg-source-v1";LOG="$PROJECT/artifacts/care-di25-roleconditioned-background-source-v1.console.log"
if tmux has-session -t "$SESSION" 2>/dev/null;then echo "tmux session already exists: $SESSION";exit 0;fi
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && bash scripts/care-di25-roleconditioned-background-source-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION";echo "Monitor: tail -F $LOG";echo "GPU logs: $PROJECT/artifacts/care-di25-roleconditioned-background-source-v1/logs/";echo "Attach: tmux attach -t $SESSION"
