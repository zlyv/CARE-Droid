#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SESSION=care-p31lr-role-direction-v1
LOG="$PWD/artifacts/care-p31lr-c43-role-direction-probe-source-v1.console.log"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "Session already exists: $SESSION"; exit 1; }
tmux new-session -d -s "$SESSION" "cd '$PWD' && bash scripts/care-p31lr-c43-role-direction-probe-source-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F artifacts/care-p31lr-c43-role-direction-probe-source-v1.console.log"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
