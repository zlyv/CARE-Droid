#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-haig
LOG=/home/boyuanjie/projects/tasa-droid/artifacts/care-p31rf-haig-source-v1.console.log
mkdir -p artifacts
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "bash scripts/care-p31rf-haig-source-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Worker logs: artifacts/care-p31rf-haig-source-v1/logs/"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
