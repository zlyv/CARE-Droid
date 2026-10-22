#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/boyuanjie/projects/tasa-droid
SESSION=care-noego-within-year-v1
LOG="$PROJECT/artifacts/care-di25-noego-within-year-replay-v1.console.log"

cd "$PROJECT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "bash scripts/care-di25-noego-within-year-replay-v1/run_on_server.sh 2>&1 | tee '$LOG'"

echo "started tmux session: $SESSION"
echo "log: $LOG"
