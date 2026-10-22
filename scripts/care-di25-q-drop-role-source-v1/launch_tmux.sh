#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../..";S="care-q-drop-role-v1";L="artifacts/care-di25-q-drop-role-source-v1.console.log"
if tmux has-session -t "$S" 2>/dev/null;then echo "tmux session already exists: $S";echo "Attach: tmux attach -t $S";exit 0;fi
tmux new-session -d -s "$S" "cd '$PWD' && bash scripts/care-di25-q-drop-role-source-v1/run_on_server.sh 2>&1 | tee '$L'"
echo "Started: $S";echo "Monitor: tail -F $L";echo "Attach: tmux attach -t $S";echo "The least-used GPU is selected automatically; override with CARE_GPU=0 or 1."
