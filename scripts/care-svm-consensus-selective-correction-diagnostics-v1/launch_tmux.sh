#!/usr/bin/env bash
set -euo pipefail
PROJECT="${SCSCD_PROJECT:-/home/boyuanjie/projects/tasa-droid}";SESSION="care-svm-consensus-diagnostics-v1"
OUTPUT="$PROJECT/outputs/care-svm-consensus-selective-correction-diagnostics-v1";CONSOLE="$PROJECT/outputs/care-svm-consensus-selective-correction-diagnostics-v1.console.log"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";mkdir -p "$PROJECT/outputs" "$OUTPUT/logs"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "Session already exists: $SESSION";exit 1; }
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && bash '$HERE/run_on_server.sh' 2>&1 | tee '$CONSOLE'"
echo "Started: $SESSION";echo "Monitor: tail -F $CONSOLE";echo "Attach: tmux attach -t $SESSION"
