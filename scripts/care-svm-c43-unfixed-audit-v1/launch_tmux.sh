#!/usr/bin/env bash
set -euo pipefail

PROJECT="${CSCUA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${CSCUA_OUTPUT:-$PROJECT/outputs/care-svm-c43-unfixed-audit-v1}"
SESSION="care-svm-c43-unfixed-audit-v1"
CONSOLE="$PROJECT/outputs/care-svm-c43-unfixed-audit-v1.console.log"

mkdir -p "$PROJECT/outputs"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION"
  exit 1
fi
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && CSCUA_OUTPUT='$OUTPUT' bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee '$CONSOLE'"
echo "Started: $SESSION"
echo "Monitor: tail -F $CONSOLE"
echo "Attach: tmux attach -t $SESSION"
echo "Expected runtime: under 1 minute; GPU is not used."
