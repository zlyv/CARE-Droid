#!/usr/bin/env bash
set -euo pipefail
PROJECT="${CMW_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${CMW_OUTPUT:-$PROJECT/outputs/care-contextual-multihop-witness-v1}"
SESSION="${CMW_SESSION:-care-contextual-multihop-witness-v1}"
mkdir -p "$OUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  exit 1
fi
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && bash '$SCRIPT_DIR/run_on_server.sh' > '$OUT.console.log' 2>&1"
echo "Started: $SESSION"
echo "Monitor: tail -F $OUT.console.log"
echo "Run log: tail -F $OUT/logs/run.log"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
