#!/usr/bin/env bash
set -euo pipefail
PROJECT="${MALEVAL_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
NAME="care-maleval2026-external-v1"
OUTPUT="${MALEVAL_OUTPUT:-$PROJECT/outputs/care-maleval2026-external-v1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$PROJECT/outputs" "$OUTPUT/logs"
if tmux has-session -t "$NAME" 2>/dev/null; then
  echo "Session already exists: $NAME"; exit 1
fi
tmux new-session -d -s "$NAME" "cd '$PROJECT' && bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee -a '$PROJECT/outputs/care-maleval2026-external-v1.console.log'"
echo "Started: $NAME"
echo "Monitor: tail -F $PROJECT/outputs/care-maleval2026-external-v1.console.log"
echo "Attach: tmux attach -t $NAME"

