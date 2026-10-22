#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
SESSION=care-cfg-witness-fixed-source-future-v2
LOG=$ROOT/outputs/care_cfg_witness_fixed_source_future_v2.console.log
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session exists: $SESSION"
  exit 1
fi
tmux new-session -d -s "$SESSION" "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 CFGW_FS_DEVICE=cuda nice -n 10 bash scripts/care-cfg-witness-fixed-source-future-v2/run.sh 2>&1 | tee -a '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
