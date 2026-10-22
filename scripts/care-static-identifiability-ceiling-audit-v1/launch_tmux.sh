#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${SICA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${SICA_OUTPUT:-$PROJECT/outputs/care-static-identifiability-ceiling-audit-v1}"
SESSION="${SICA_SESSION:-care-static-identifiability-v1}"
CONSOLE="$PROJECT/outputs/care-static-identifiability-ceiling-audit-v1.console.log"
mkdir -p "$PROJECT/outputs"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Session already exists: $SESSION" >&2; exit 1; fi
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && SICA_PROJECT='$PROJECT' SICA_OUTPUT='$OUTPUT' bash '$HERE/run_on_server.sh' 2>&1 | tee -a '$CONSOLE'"
echo "Started: $SESSION"
echo "Output: $OUTPUT"
echo "Monitor: tail -F $CONSOLE"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
