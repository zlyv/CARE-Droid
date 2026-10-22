#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-role-gate
LOG="$PWD/artifacts/care-role-reliability-gate-v1.console.log"
mkdir -p artifacts/care-role-reliability-gate-v1
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "bash scripts/care-role-reliability-gate-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Nested logs: artifacts/care-role-reliability-gate-v1/logs/"
echo "Attach: tmux attach -t $SESSION"

