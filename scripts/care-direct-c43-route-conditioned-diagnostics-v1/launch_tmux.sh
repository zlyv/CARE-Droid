#!/usr/bin/env bash
set -euo pipefail
PROJECT="${RCTD_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="care-direct-c43-route-conditioned-diagnostics-v1"
SESSION="care-c43-route-diagnostics-v1"
CONSOLE="$PROJECT/outputs/$NAME.console.log"
mkdir -p "$PROJECT/outputs"
if [ -e "$PROJECT/outputs/$NAME" ] || [ -e "$CONSOLE" ]; then
  echo "Existing diagnostic output/log: use a new version."
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION"
  exit 1
fi
export RCTD_GPU="${RCTD_GPU:-1}"
tmux new-session -d -s "$SESSION" "bash '$SCRIPT_DIR/run_on_server.sh' > '$CONSOLE' 2>&1"
echo "Started: $SESSION (GPU $RCTD_GPU)"
echo "Monitor: tail -F $CONSOLE"
echo "Attach: tmux attach -t $SESSION"
