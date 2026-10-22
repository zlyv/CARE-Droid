#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-formal-v1
A="$PWD/artifacts/care-droid-formal-v1"
LOG="$PWD/artifacts/care-droid-formal-v1.console.log"
mkdir -p "$A"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "bash scripts/care-droid-formal-v1/run_care_droid_formal_v1.sh 2>&1 | tee '$LOG' '$A/run.log'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"

