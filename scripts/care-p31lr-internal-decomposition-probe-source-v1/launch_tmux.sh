#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-p31lr-internal-decomp-v1
LOG="$PWD/artifacts/care-p31lr-internal-decomposition-probe-source-v1.console.log"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "Session exists: $SESSION"; exit 1; }
tmux new-session -d -s "$SESSION" "cd '$PWD' && bash scripts/care-p31lr-internal-decomposition-probe-source-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F artifacts/care-p31lr-internal-decomposition-probe-source-v1.console.log"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
