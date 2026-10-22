#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-role-api;LOG=artifacts/care-role-conditioned-api-v1.console.log;RUNLOG=artifacts/care-role-conditioned-api-v1/run.log
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "bash scripts/care-role-conditioned-api-v1/run_on_server.sh 2>&1 | tee '$RUNLOG' >'$LOG'"
echo "Started: $SESSION";echo "Monitor: tail -F $LOG";echo "GPU logs: artifacts/care-role-conditioned-api-v1/logs/";echo "Attach: tmux attach -t $SESSION"
