#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-role-direction
LOG="$PWD/artifacts/care-class-conditional-role-direction-v1.console.log"
A="$PWD/artifacts/care-class-conditional-role-direction-v1"
mkdir -p "$A"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "bash scripts/care-class-conditional-role-direction-v1/run_on_server.sh 2>&1 | tee '$LOG' '$A/run.log'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
