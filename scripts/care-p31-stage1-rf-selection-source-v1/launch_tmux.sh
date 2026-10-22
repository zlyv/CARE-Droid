#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-p31-rf-selection-v1
LOG="$PWD/artifacts/care-p31-stage1-rf-selection-source-v1.console.log"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "Session exists: $SESSION"; exit 1; }
tmux new-session -d -s "$SESSION" "cd '$PWD' && bash scripts/care-p31-stage1-rf-selection-source-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F artifacts/care-p31-stage1-rf-selection-source-v1.console.log"
echo "GPU logs: artifacts/care-p31-stage1-rf-selection-source-v1/logs/"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
