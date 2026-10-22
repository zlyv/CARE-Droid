#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${SGCA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${SGCA_OUTPUT:-$PROJECT/outputs/care-sensitive-guard-context-audit-v1}"
SESSION="${SGCA_SESSION:-care-sensitive-guard-audit-v1}"
CONSOLE="$PROJECT/outputs/care-sensitive-guard-context-audit-v1.console.log"

mkdir -p "$PROJECT/outputs"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && SGCA_PROJECT='$PROJECT' SGCA_OUTPUT='$OUTPUT' bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee -a '$CONSOLE'"
echo "Started: $SESSION"
echo "Output: $OUTPUT"
echo "Monitor: tail -F $CONSOLE"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
