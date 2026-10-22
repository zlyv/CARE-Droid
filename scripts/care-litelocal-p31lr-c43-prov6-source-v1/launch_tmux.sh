#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-p31lr-c43-prov6-v1
LOG="$PWD/artifacts/care-litelocal-p31lr-c43-prov6-source-v1.console.log"
mkdir -p "$(dirname "$LOG")"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "Already running: $SESSION"; exit 0; }
tmux new-session -d -s "$SESSION" "cd '$PWD' && bash scripts/care-litelocal-p31lr-c43-prov6-source-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "GPU logs: artifacts/care-litelocal-p31lr-c43-prov6-source-v1/logs/"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
