#!/usr/bin/env bash
set -euo pipefail
PROJECT="${SVMPBE_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${SVMPBE_OUTPUT:-$PROJECT/outputs/care-svm-positive-benign-evidence-v1}"
SESSION="${SVMPBE_SESSION:-care-svm-positive-benign-v1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$PROJECT/outputs"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Session already exists: $SESSION"; exit 1; fi
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee '$OUTPUT.console.log'"
echo "Started: $SESSION"
echo "Monitor: tail -F $OUTPUT.console.log"
echo "Run log: tail -F $OUTPUT/logs/run.log"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"

