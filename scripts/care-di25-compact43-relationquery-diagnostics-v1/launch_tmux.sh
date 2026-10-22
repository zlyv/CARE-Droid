#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION="care-compact43-relationquery-diag-v1"
LOG="artifacts/care-di25-compact43-relationquery-diagnostics-v1.console.log"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Already running: $SESSION"; exit 0; fi
tmux new-session -d -s "$SESSION" "cd /home/boyuanjie/projects/tasa-droid && bash scripts/care-di25-compact43-relationquery-diagnostics-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "GPU0 frozen log: artifacts/care-di25-compact43-relationquery-diagnostics-v1/logs/frozen_C43_RELATION_QUERY.gpu0.log"
echo "GPU1 within log: artifacts/care-di25-compact43-relationquery-diagnostics-v1/logs/within_C43_RELATION_QUERY.gpu1.log"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
