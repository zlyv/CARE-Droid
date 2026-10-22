#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-p31-lr-robust-2021-quick
CONSOLE=/home/boyuanjie/projects/tasa-droid/outputs/care-p31-lr-robust-2021-quick.console.log
PY="${PLRRS_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
mkdir -p outputs/care-p31-lr-robust-static-v1
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION"
  exit 1
fi
tmux new-session -d -s "$SESSION" "$PY scripts/care-p31-lr-robust-static-v1/quick_2021_stage1.py 2>&1 | tee '$CONSOLE'"
echo "Started: $SESSION"
echo "Monitor: tail -F $CONSOLE"
echo "Completion: CARE_P31_LR_ROBUST_QUICK_2021_DONE"
