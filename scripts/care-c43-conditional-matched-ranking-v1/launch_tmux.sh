#!/usr/bin/env bash
set -euo pipefail
PROJECT="${CCMR_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SESSION="care-c43-conditional-matched-v1"
OUTPUT="${CCMR_OUTPUT:-$PROJECT/outputs/care-c43-conditional-matched-ranking-v1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$PROJECT/outputs" "$OUTPUT/logs"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "Already running: $SESSION"; exit 1; }
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee '$PROJECT/outputs/care-c43-conditional-matched-ranking-v1.console.log'"
echo "Started: $SESSION"
echo "Monitor: tail -F $PROJECT/outputs/care-c43-conditional-matched-ranking-v1.console.log"
echo "Attach: tmux attach -t $SESSION"
