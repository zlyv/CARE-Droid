#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
SESSION=care-cfg-rolegraph-witness-all-years-v2
LOG=$ROOT/outputs/care_cfg_rolegraph_witness_all_years_v2.console.log
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "session exists" >&2; exit 1; fi
if [ -e "$ROOT/artifacts/care_cfg_witness_all_years_v2" ] || [ -e "$LOG" ]; then echo "new output already exists; refusing overwrite" >&2; exit 1; fi
tmux new-session -d -s "$SESSION" "cd '$ROOT' && CFGW_WORKERS='${CFGW_WORKERS:-4}' bash scripts/care-host-cfg-rolegraph-witness-all-years-v2/run.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Attach: tmux attach -t $SESSION"
echo "Collect: bash scripts/care-host-cfg-rolegraph-witness-all-years-v2/collect_results.sh"

