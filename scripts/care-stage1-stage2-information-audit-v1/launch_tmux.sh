#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/boyuanjie/projects/tasa-droid
SESSION=care-information-audit-v1
SCRIPT="$PROJECT/scripts/care-stage1-stage2-information-audit-v1"
CONSOLE="$PROJECT/outputs/care-stage1-stage2-information-audit-v1.console.log"

cd "$PROJECT"
mkdir -p "$PROJECT/outputs"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "BLOCKED_BY_CONTRACT: tmux session already exists: $SESSION"
  exit 1
fi
if [[ -e "$PROJECT/outputs/care-stage1-stage2-information-audit-v1" ]]; then
  echo "BLOCKED_BY_CONTRACT: output already exists; use a new version instead of overwriting"
  exit 1
fi
tmux new-session -d -s "$SESSION" "bash '$SCRIPT/run_on_server.sh' > '$CONSOLE' 2>&1"
echo "Started: $SESSION"
echo "Monitor: tail -F $CONSOLE"
echo "Attach: tmux attach -t $SESSION"

