#!/usr/bin/env bash
set -euo pipefail
PROJECT="${RCT_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="care-direct-c43-route-conditioned-training-v2"
SESSION="care-c43-route-training-v2"
CONSOLE="$PROJECT/outputs/$NAME.console.log"
mkdir -p "$PROJECT/outputs"
if [ -e "$PROJECT/outputs/$NAME" ] || [ -e "$CONSOLE" ]; then
  echo "Experiment output/log exists: use a new version; no overwrite."
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session exists: $SESSION"
  exit 1
fi
export RCT_GPU="${RCT_GPU:-1}"
tmux new-session -d -s "$SESSION" "bash '$SCRIPT_DIR/run_on_server.sh' > '$CONSOLE' 2>&1"
echo "Started: $SESSION (GPU $RCT_GPU)"
echo "Monitor: tail -F $CONSOLE"
echo "Results: bash $SCRIPT_DIR/collect_results.sh"
echo "Attach: tmux attach -t $SESSION"
