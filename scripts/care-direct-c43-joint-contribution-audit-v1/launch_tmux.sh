#!/usr/bin/env bash
set -euo pipefail
PROJECT="${JCIA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${JCIA_OUTPUT:-$PROJECT/outputs/care-direct-c43-joint-contribution-audit-v1}"
SESSION="${JCIA_SESSION:-care-direct-c43-joint-audit-v1}"
CONSOLE="$PROJECT/outputs/care-direct-c43-joint-contribution-audit-v1.console.log"
mkdir -p "$PROJECT/outputs"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Session exists: $SESSION"; exit 1; fi
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee -a '$CONSOLE'"
echo "Started: $SESSION"
echo "Monitor: tail -F $CONSOLE"
echo "Worker log: tail -F $OUTPUT/logs/audit.log"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"

