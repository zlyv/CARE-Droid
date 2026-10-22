#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/boyuanjie/projects/tasa-droid
SESSION=care-group-compact-validation-v1
SCRIPT="$PROJECT/scripts/care-group-calibration-compact43-validation-v1"
CONSOLE="$PROJECT/outputs/care-group-calibration-compact43-validation-v1.console.log"

cd "$PROJECT"
mkdir -p "$PROJECT/outputs"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "BLOCKED_BY_CONTRACT: tmux session already exists: $SESSION"
  exit 1
fi
if [[ -e "$PROJECT/outputs/care-group-calibration-compact43-validation-v1" ]]; then
  echo "BLOCKED_BY_CONTRACT: output exists; use a new version"
  exit 1
fi
tmux new-session -d -s "$SESSION" "bash '$SCRIPT/run_on_server.sh' > '$CONSOLE' 2>&1"
echo "Started: $SESSION"
echo "Monitor: tail -F $CONSOLE"
echo "Attach: tmux attach -t $SESSION"

