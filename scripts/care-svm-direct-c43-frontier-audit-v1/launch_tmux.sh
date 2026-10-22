#!/usr/bin/env bash
set -euo pipefail
PROJECT="${DCFA_PROJECT:-/home/boyuanjie/projects/tasa-droid}";SESSION="care-svm-direct-c43-frontier-v1";OUTPUT="${DCFA_OUTPUT:-$PROJECT/outputs/care-svm-direct-c43-frontier-audit-v1}";SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$PROJECT/outputs" "$OUTPUT";tmux has-session -t "$SESSION" 2>/dev/null && { echo "Already running: $SESSION";exit 1; }
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee '$PROJECT/outputs/care-svm-direct-c43-frontier-audit-v1.console.log'"
echo "Started: $SESSION";echo "Monitor: tail -F $PROJECT/outputs/care-svm-direct-c43-frontier-audit-v1.console.log";echo "Attach: tmux attach -t $SESSION"
