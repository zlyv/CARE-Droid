#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${BRF_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${BRF_OUTPUT:-$PROJECT/outputs/care-svm-direct-c43-budget-frontier-v1}"
SESSION="${BRF_SESSION:-care-svm-direct-c43-budget-frontier-v1}"
CONSOLE="$PROJECT/outputs/care-svm-direct-c43-budget-frontier-v1.console.log"
mkdir -p "$PROJECT/outputs" "$OUTPUT/logs"; touch "$CONSOLE"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "Session already exists: $SESSION"; exit 1; }
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && BRF_PROJECT='$PROJECT' BRF_OUTPUT='$OUTPUT' BRF_PYTHON='${BRF_PYTHON:-}' BRF_PREDICTIONS='${BRF_PREDICTIONS:-}' BRF_ARCHITECTURE='${BRF_ARCHITECTURE:-}' bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee -a '$CONSOLE'"
echo "Started: $SESSION"
echo "Monitor: tail -F $CONSOLE"
echo "Collect: bash $SCRIPT_DIR/collect_results.sh"

