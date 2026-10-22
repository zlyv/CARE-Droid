#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT="${TDA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"; OUTPUT="${TDA_OUTPUT:-$PROJECT/outputs/care-direct-c43-temporal-direction-audit-v1}"; SESSION="care-c43-temporal-direction-v1"
mkdir -p "$PROJECT/outputs"; tmux new-session -d -s "$SESSION" "cd '$PROJECT' && bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee '$OUTPUT.console.log'"
echo "Started: $SESSION"; echo "Monitor: tail -F $OUTPUT.console.log"; echo "Attach: tmux attach -t $SESSION"
