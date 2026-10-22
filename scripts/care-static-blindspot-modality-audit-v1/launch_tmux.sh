#!/usr/bin/env bash
set -euo pipefail
PROJECT="${SBMA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${SBMA_OUTPUT:-$PROJECT/outputs/care-static-blindspot-modality-audit-v1}"
SESSION="care-static-blindspot-v1";CONSOLE="$PROJECT/outputs/care-static-blindspot-modality-audit-v1.console.log"
mkdir -p "$PROJECT/outputs" "$OUTPUT"
if tmux has-session -t "$SESSION" 2>/dev/null;then echo "Session already exists: $SESSION";exit 1;fi
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee -a '$CONSOLE'"
echo "Started: $SESSION"
echo "Monitor: tail -F $CONSOLE"
echo "Run log: tail -F $OUTPUT/logs/run.log"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
