#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION="care-future-assets-v2"
LOG="/home/boyuanjie/projects/tasa-droid/artifacts/care-future-local-assets-v2.console.log"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2; exit 1
fi
mkdir -p artifacts
tmux new-session -d -s "$SESSION" \
  "cd /home/boyuanjie/projects/tasa-droid && while tmux has-session -t care-source-assets-v2 2>/dev/null; do echo '[queue] waiting for care-source-assets-v2 to finish'; sleep 60; done; bash scripts/care-future-local-assets-v2/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Queued after care-source-assets-v2: $SESSION"
echo "Monitor queue: tmux capture-pane -p -S -50 -t $SESSION"
echo "Monitor extraction: tail -F $LOG"
