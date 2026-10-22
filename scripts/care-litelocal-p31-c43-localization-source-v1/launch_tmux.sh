#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid

SESSION=care-p31-c43-localization-v1
CONSOLE=/home/boyuanjie/projects/tasa-droid/artifacts/care-litelocal-p31-c43-localization-source-v1.console.log
mkdir -p artifacts
: > "$CONSOLE"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Already running: $SESSION"
  exit 0
fi
tmux new-session -d -s "$SESSION" \
  "cd /home/boyuanjie/projects/tasa-droid && stdbuf -oL -eL bash scripts/care-litelocal-p31-c43-localization-source-v1/run_on_server.sh 2>&1 | tee '$CONSOLE'"
echo "Started: $SESSION"
echo "Monitor: tail -F artifacts/care-litelocal-p31-c43-localization-source-v1.console.log"
echo "GPU logs: artifacts/care-litelocal-p31-c43-localization-source-v1/logs/"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
