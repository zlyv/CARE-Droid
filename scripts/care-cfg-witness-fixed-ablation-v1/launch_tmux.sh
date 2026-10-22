#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
SESSION=care-cfg-witness-fixed-ablation-v1
LOG=$ROOT/outputs/care_cfg_witness_fixed_ablation_v1.console.log
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "session exists" >&2; exit 1; fi
if [ -e "$ROOT/outputs/care_cfg_witness_fixed_ablation_v1" ] || [ -e "$LOG" ]; then echo "output exists; refusing overwrite" >&2; exit 1; fi
tmux new-session -d -s "$SESSION" "cd '$ROOT' && CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-1}' bash scripts/care-cfg-witness-fixed-ablation-v1/run.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Attach: tmux attach -t $SESSION"
echo "Collect: bash scripts/care-cfg-witness-fixed-ablation-v1/collect_results.sh"
