#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-host-size-control-v1
LOG=artifacts/care-host-complement-size-controlled-audit-v1.console.log
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "tmux session already exists: $SESSION"; echo "Attach: tmux attach -t $SESSION"; exit 0; fi
mkdir -p artifacts
tmux new-session -d -s "$SESSION" "cd /home/boyuanjie/projects/tasa-droid && bash scripts/care-host-complement-size-controlled-audit-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION";echo "Monitor: tail -F $LOG";echo "Attach: tmux attach -t $SESSION"

