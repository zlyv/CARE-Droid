#!/usr/bin/env bash
set -euo pipefail
PROJECT="${CSEF_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT="$PROJECT/scripts/care-c43-static-evidence-fidelity-audit-v1"
OUTPUT="${CSEF_OUTPUT:-$PROJECT/outputs/care-c43-static-evidence-fidelity-audit-v1}"
SESSION="care-c43-evidence-fidelity-v1"
CONSOLE="$PROJECT/outputs/care-c43-static-evidence-fidelity-audit-v1.console.log"
mkdir -p "$PROJECT/outputs" "$OUTPUT/logs"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Session already exists: $SESSION"; exit 1; fi
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && CSEF_OUTPUT='$OUTPUT' bash '$SCRIPT/run_on_server.sh' 2>&1 | tee -a '$CONSOLE'"
echo "Started: $SESSION"
echo "Monitor: tail -F $CONSOLE"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
