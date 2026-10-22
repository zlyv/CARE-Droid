#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-p31-lr-l6-only-v1
CONSOLE=/home/boyuanjie/projects/tasa-droid/outputs/care-p31-lr-robust-static-l6-v1.console.log
GPU="${PLRRS_GPU:-1}"
mkdir -p outputs
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION"
  exit 1
fi
tmux new-session -d -s "$SESSION" "PLRRS_GPU='$GPU' bash scripts/care-p31-lr-robust-static-v1/run_l6_only.sh 2>&1 | tee '$CONSOLE'"
echo "Started: $SESSION"
echo "GPU: $GPU"
echo "Monitor: tail -F $CONSOLE"
echo "Worker logs: tail -F outputs/care-p31-lr-robust-static-l6-v1/logs/*.log"
echo "Attach: tmux attach -t $SESSION"
echo "Completion: CARE_P31_LR_L6_ONLY_PIPELINE_DONE"
