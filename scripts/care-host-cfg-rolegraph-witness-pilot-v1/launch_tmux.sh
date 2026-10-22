#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
SESSION=care-cfg-rolegraph-witness-pilot-v1
LOG=$ROOT/outputs/care_cfg_rolegraph_witness_pilot_2017.console.log
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "session exists" >&2; exit 1; fi
if [ -e "$ROOT/artifacts/care_cfg_witness_pilot_2017" ] || [ -e "$LOG" ]; then echo "new output already exists; refusing overwrite" >&2; exit 1; fi
tmux new-session -d -s "$SESSION" "cd '$ROOT' && CFGW_WORKERS='${CFGW_WORKERS:-4}' bash scripts/care-host-cfg-rolegraph-witness-pilot-v1/run.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Attach: tmux attach -t $SESSION"

