#!/usr/bin/env bash
set -euo pipefail
PROJECT="${CMW_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${CMW_POSTGATE_OUTPUT:-$PROJECT/outputs/care-cmw-postgate-source-audit-v1}"
SESSION="${CMW_POSTGATE_SESSION:-care-cmw-postgate-source-audit-v1}"
mkdir -p "$OUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION"; exit 1
fi
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && bash '$SCRIPT_DIR/run_on_server.sh' > '$OUT.console.log' 2>&1"
echo "Started: $SESSION"
echo "Monitor: tail -F $OUT.console.log"
echo "Attach: tmux attach -t $SESSION"
