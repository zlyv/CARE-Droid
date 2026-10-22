#!/usr/bin/env bash
set -euo pipefail

PROJECT="${CARE_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${CARE_LINEAGE_OUTPUT:-${PROJECT}/outputs/care-p31-lineage-host-audit-v1}"
SESSION="care-p31-lineage-host-audit-v1"
CONSOLE="${PROJECT}/outputs/care-p31-lineage-host-audit-v1.console.log"

mkdir -p "$PROJECT/outputs" "$OUTPUT/logs"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee -a '$CONSOLE'"
echo "Started: $SESSION"
echo "Output: $OUTPUT"
echo "Monitor: tail -F $CONSOLE"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
