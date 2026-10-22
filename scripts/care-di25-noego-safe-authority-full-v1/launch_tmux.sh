#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SESSION=care-safe-authority-full-v1
LOG=artifacts/care-di25-noego-safe-authority-full-v1.console.log
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "tmux session already exists: $SESSION";exit 2;fi
mkdir -p artifacts
tmux new-session -d -s "$SESSION" "cd '$PWD' && CUDA_VISIBLE_DEVICES=1 bash scripts/care-di25-noego-safe-authority-full-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION";echo "Monitor: tail -F $LOG";echo "Attach: tmux attach -t $SESSION"

