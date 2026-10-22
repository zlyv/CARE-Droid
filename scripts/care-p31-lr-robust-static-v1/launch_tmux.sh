#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-p31-lr-robust-static-v1
CONSOLE=/home/boyuanjie/projects/tasa-droid/outputs/care-p31-lr-robust-static-v1.console.log
GPU_PLAN="${PLRRS_GPUS:-1}"
mkdir -p /home/boyuanjie/projects/tasa-droid/outputs
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION"
  exit 1
fi
tmux new-session -d -s "$SESSION" "PLRRS_GPUS='$GPU_PLAN' bash scripts/care-p31-lr-robust-static-v1/run_on_server.sh 2>&1 | tee '$CONSOLE'"
echo "Started: $SESSION"
echo "GPU plan: $GPU_PLAN (set PLRRS_GPUS=0,1 before launch only when both GPUs are available)"
echo "Monitor: tail -F $CONSOLE"
echo "Worker logs: tail -F outputs/care-p31-lr-robust-static-v1/logs/*.log"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
