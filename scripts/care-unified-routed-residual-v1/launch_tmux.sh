#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-unified-routed-residual-v1
LOG=/home/boyuanjie/projects/tasa-droid/outputs/care-unified-routed-residual-v1.console.log
mkdir -p outputs
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION"; exit 1
fi
tmux new-session -d -s "$SESSION" "bash scripts/care-unified-routed-residual-v1/run_on_server.sh > '$LOG' 2>&1"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Collect: bash scripts/care-unified-routed-residual-v1/collect_results.sh"
