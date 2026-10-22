#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PASEA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${PASEA_OUTPUT:-$PROJECT/outputs/care-pasea-audit-v1}"
SESSION="${PASEA_TMUX_SESSION:-care-pasea-audit-v1}"
CONSOLE="${PASEA_CONSOLE_LOG:-$PROJECT/outputs/care-pasea-audit-v1.console.log}"
mkdir -p "$PROJECT/outputs" "$OUTPUT/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" "cd '$PROJECT' && bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee -a '$CONSOLE' '$OUTPUT/logs/run.log'"
echo "Started: $SESSION"
echo "Monitor: tail -F $CONSOLE"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"

