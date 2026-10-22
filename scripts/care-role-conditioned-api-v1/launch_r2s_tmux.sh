#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-role-api-r2s;LOG=artifacts/care-role-conditioned-api-v1.r2s.console.log
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "bash scripts/care-role-conditioned-api-v1/run_r2s_control.sh >'$LOG' 2>&1"
echo "Started: $SESSION";echo "Monitor: tail -F $LOG";echo "GPU log: artifacts/care-role-conditioned-api-v1/logs/R2S_FIXED_SHUFFLED.gpu0.log"
