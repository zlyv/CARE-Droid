#!/usr/bin/env bash
set -euo pipefail

PROJECT="${OBCBC_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OBCBC_OUTPUT:-$PROJECT/outputs/care-obcbc-source-v1}"
SESSION="${OBCBC_TMUX_SESSION:-care-obcbc-static-v1}"
CONSOLE="$PROJECT/outputs/care-obcbc-source-v1.console.log"

mkdir -p "$PROJECT/outputs" "$OUT/logs"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "cd '$PROJECT' && bash '$SCRIPT_DIR/run_on_server.sh' >> '$CONSOLE' 2>&1"
echo "Started: $SESSION"
echo "Output: $OUT"
echo "Monitor: tail -F $CONSOLE"
echo "Run log: tail -F $OUT/logs/run.log"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
