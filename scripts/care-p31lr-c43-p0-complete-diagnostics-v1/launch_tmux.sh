#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-p31lr-c43-p0-complete-v1
LOG="$PWD/artifacts/care-p31lr-c43-p0-complete-diagnostics-v1.console.log"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "Session already exists: $SESSION";exit 1; }
tmux new-session -d -s "$SESSION" "cd '$PWD' && bash scripts/care-p31lr-c43-p0-complete-diagnostics-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F artifacts/care-p31lr-c43-p0-complete-diagnostics-v1.console.log"
echo "Frozen log: artifacts/care-p31lr-c43-p0-complete-diagnostics-v1/logs/frozen.gpu0.log"
echo "Within logs: artifacts/care-p31lr-c43-p0-complete-diagnostics-v1/logs/"
echo "Attach: tmux attach -t $SESSION"
