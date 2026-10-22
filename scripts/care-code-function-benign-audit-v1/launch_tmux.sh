#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-code-function-benign-audit-v1
LOG=/home/boyuanjie/projects/tasa-droid/outputs/care-code-function-benign-audit-v1.console.log
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session exists: $SESSION"; exit 1
fi
mkdir -p outputs
tmux new-session -d -s "$SESSION" "bash scripts/care-code-function-benign-audit-v1/run_on_server.sh > '$LOG' 2>&1"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Collect: bash scripts/care-code-function-benign-audit-v1/collect_results.sh"
