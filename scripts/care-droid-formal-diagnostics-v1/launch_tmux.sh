#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-formal-diagnostics-v1
LOG="$PWD/artifacts/care-droid-formal-diagnostics-v1.console.log"
mkdir -p artifacts/care-droid-formal-diagnostics-v1
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "bash scripts/care-droid-formal-diagnostics-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Frozen GPU0: tail -F artifacts/care-droid-formal-diagnostics-v1/logs/frozen.gpu0.log"
echo "Within GPU1: tail -F artifacts/care-droid-formal-diagnostics-v1/logs/within_2018_2019.gpu1.log"
echo "Within GPU0 later: tail -F artifacts/care-droid-formal-diagnostics-v1/logs/within_2020_2021.gpu0.log"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
