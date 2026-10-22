#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-viewdi-source-v1
LOG="$PWD/artifacts/care-viewdi-router-source-v1.console.log"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"; exit 1
fi
mkdir -p artifacts
tmux new-session -d -s "$SESSION" "cd '$PWD' && bash scripts/care-viewdi-router-source-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F artifacts/care-viewdi-router-source-v1.console.log"
echo "Attach: tmux attach -t $SESSION"

