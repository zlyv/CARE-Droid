#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${EMO_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${EMO_OUTPUT:-$PROJECT/outputs/care-svm-elastic-meanopposition-v1}"
SESSION="care-svm-elastic-meanopposition-v1"
mkdir -p "$PROJECT/outputs"
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee '$OUTPUT.console.log'"
echo "Started: $SESSION"
echo "Monitor: tail -F $OUTPUT.console.log"
echo "Attach: tmux attach -t $SESSION"
