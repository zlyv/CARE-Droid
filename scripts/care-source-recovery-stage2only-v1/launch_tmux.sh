#!/usr/bin/env bash
set -euo pipefail

cd /home/boyuanjie/projects/tasa-droid
SESSION="${SESSION:-care-stage2only-v1}"
LOG="artifacts/care-source-recovery-stage2only-v1.console.log"
mkdir -p artifacts

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd /home/boyuanjie/projects/tasa-droid && bash scripts/care-source-recovery-stage2only-v1/run_full_source_on_server.sh 2>&1 | tee '$LOG'"

echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
echo "Log: tail -f $LOG"
