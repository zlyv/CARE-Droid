#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
GPU=${1:-1}
SESSION=care-p31-c43-future-v1
LOG=artifacts/care-litelocal-p31-c43-future-v1.console.log
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "cd /home/boyuanjie/projects/tasa-droid && CUDA_VISIBLE_DEVICES='$GPU' bash scripts/care-litelocal-p31-c43-future-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION on physical GPU $GPU"
echo "Monitor: tail -F $LOG"
echo "Attach: tmux attach -t $SESSION"
