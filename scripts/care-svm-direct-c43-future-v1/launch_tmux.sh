#!/usr/bin/env bash
set -euo pipefail
PROJECT="${SDC43F_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${SDC43F_OUTPUT:-$PROJECT/outputs/care-svm-direct-c43-future-v1}"
SESSION="care-svm-direct-c43-future-v1"
CONSOLE="$PROJECT/outputs/care-svm-direct-c43-future-v1.console.log"
mkdir -p "$PROJECT/outputs"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Session already exists: $SESSION"; exit 1; fi
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && SDC43F_PROJECT='$PROJECT' SDC43F_OUTPUT='$OUTPUT' bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee -a '$CONSOLE'"
echo "Started: $SESSION"
echo "Monitor: tail -F $CONSOLE"
echo "Worker log: tail -F $OUTPUT/logs/future.gpu1.log"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
