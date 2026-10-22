#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
DEF="${1:-MEAN_COMMITTEE}"
GPU="${2:-1}"
SESSION="litelocal-m0-${DEF,,}"
LOG="artifacts/care-litelocal-di25-host-m0-source-v1/${DEF}.console.log"
mkdir -p "$(dirname "$LOG")"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Already running: $SESSION"
  exit 0
fi
tmux new-session -d -s "$SESSION" \
  "cd /home/boyuanjie/projects/tasa-droid && bash scripts/care-litelocal-di25-host-m0-source-v1/run_one.sh '$DEF' '$GPU' 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Attach: tmux attach -t $SESSION"
