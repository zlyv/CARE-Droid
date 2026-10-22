#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
S=care-function-role-explanation-v1;L=/home/boyuanjie/projects/tasa-droid/outputs/care-function-role-explanation-v1.console.log
tmux has-session -t "$S" 2>/dev/null && { echo "Session exists: $S";exit 1; }
mkdir -p outputs;tmux new-session -d -s "$S" "bash scripts/care-function-role-explanation-v1/run_on_server.sh > '$L' 2>&1"
echo "Started: $S";echo "Monitor: tail -F $L";echo "Collect: bash scripts/care-function-role-explanation-v1/collect_results.sh"
