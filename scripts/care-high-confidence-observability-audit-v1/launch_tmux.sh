#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${HCOA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${HCOA_OUTPUT:-$PROJECT/outputs/care-high-confidence-observability-audit-v1}"
SESSION="${HCOA_SESSION:-care-hc-observability-v1}"
CONSOLE="$PROJECT/outputs/care-high-confidence-observability-audit-v1.console.log"
mkdir -p "$PROJECT/outputs"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Session already exists: $SESSION" >&2; exit 1; fi
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && HCOA_PROJECT='$PROJECT' HCOA_OUTPUT='$OUTPUT' bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee -a '$CONSOLE'"
echo "Started: $SESSION"
echo "Output: $OUTPUT"
echo "Monitor: tail -F $CONSOLE"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
