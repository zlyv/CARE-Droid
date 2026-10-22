#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
SESSION=care-az-random15k-available-extraction-v1
LOG="$ROOT/outputs/care_az_random15k_available_extraction_v1.console.log"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "session exists" >&2; exit 2; fi
if [ -e "$ROOT/outputs/care_az_random15k_available_extraction_v1" ] || [ -e "$LOG" ]; then
  echo "output exists; refusing overwrite" >&2; exit 2
fi
tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash scripts/care-az-random15k-available-extraction-v1/run.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"

