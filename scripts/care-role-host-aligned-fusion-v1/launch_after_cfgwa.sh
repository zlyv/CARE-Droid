#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
SESSION=care-role-host-aligned-fusion-v1
LOG=$ROOT/outputs/care_role_host_aligned_fusion_v1.console.log
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Session exists: $SESSION"; exit 1; fi
tmux new-session -d -s "$SESSION" "cd '$ROOT' && while tmux has-session -t care-cfg-witness-fixed-ablation-v1 2>/dev/null; do sleep 60; done; CUDA_VISIBLE_DEVICES=1 nice -n 10 bash scripts/care-role-host-aligned-fusion-v1/run.sh 2>&1 | tee -a '$LOG'"
echo "Queued: $SESSION"
echo "Monitor: tail -F $LOG"
echo "The run starts automatically after care-cfg-witness-fixed-ablation-v1 exits."
