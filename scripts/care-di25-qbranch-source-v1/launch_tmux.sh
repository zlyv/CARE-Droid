#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SESSION="care-qbranch-source-v1"
LOG="artifacts/care-di25-qbranch-source-v1.console.log"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi
tmux new-session -d -s "$SESSION" \
  "cd '$PWD' && bash scripts/care-di25-qbranch-source-v1/run_parallel_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "GPU0 log: tail -F artifacts/care-di25-qbranch-source-v1/logs/gpu0.log"
echo "GPU1 log: tail -F artifacts/care-di25-qbranch-source-v1/logs/gpu1.log"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
