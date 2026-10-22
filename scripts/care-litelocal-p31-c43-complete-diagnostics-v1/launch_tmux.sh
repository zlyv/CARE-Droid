#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-p31-c43-complete-v1
LOG=artifacts/care-litelocal-p31-c43-complete-diagnostics-v1.console.log
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "cd /home/boyuanjie/projects/tasa-droid && bash scripts/care-litelocal-p31-c43-complete-diagnostics-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "GPU logs: artifacts/care-litelocal-p31-c43-complete-diagnostics-v1/logs/"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
