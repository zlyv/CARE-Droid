#!/usr/bin/env bash
# Launch the isolated SCBE pipeline in a detached tmux session; no existing task is stopped.
set -euo pipefail
ROOT=/home/boyuanjie/projects/tasa-droid
OUT="$ROOT/outputs/care_scbe_v1"
SESSION=care-scbe-v1
cd "$ROOT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Existing tmux session: $SESSION" >&2;exit 2
fi
LOG=console.log
if [[ -e "$OUT/$LOG" ]]; then LOG=console.retry1.log; fi
if [[ -e "$OUT/$LOG" ]]; then
  echo 'Both console logs exist; refusing overwrite or append to previous run.' >&2;exit 2
fi
if [[ ! -f "$OUT/scbe_interface_validation.json" ]]; then
  echo 'SCBE interface smoke test has not passed.' >&2;exit 2
fi
tmux new-session -d -s "$SESSION" "bash scripts/care-scbe-v1/pipeline.sh > outputs/care_scbe_v1/$LOG 2>&1"
echo "Started: $SESSION"
echo "Monitor: tail -F $OUT/$LOG"
