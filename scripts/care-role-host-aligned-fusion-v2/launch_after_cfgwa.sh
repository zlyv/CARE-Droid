#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
SESSION=care-role-host-aligned-fusion-v2
LOG=$ROOT/outputs/care_role_host_aligned_fusion_v2.console.log
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Session exists: $SESSION"; exit 1; fi
if test -e "$ROOT/outputs/care_role_host_aligned_fusion_v2" || test -e "$LOG"; then
  echo "Refusing to overwrite existing v2 output or console log"; exit 1
fi
tmux new-session -d -s "$SESSION" "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 nice -n 10 bash scripts/care-role-host-aligned-fusion-v2/run.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
