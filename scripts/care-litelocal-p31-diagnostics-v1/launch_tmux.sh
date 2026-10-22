#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION="care-litelocal-p31-diagnostics-v1"
LOG="artifacts/care-litelocal-p31-diagnostics-v1.console.log"
if tmux has-session -t "$SESSION" 2>/dev/null;then echo "Already running: $SESSION";exit 0;fi
tmux new-session -d -s "$SESSION" "cd /home/boyuanjie/projects/tasa-droid && bash scripts/care-litelocal-p31-diagnostics-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Extraction shards: artifacts/care-litelocal-p31-diagnostics-v1/future_p31_shards/"
echo "Frozen job log: artifacts/care-litelocal-p31-diagnostics-v1/frozen_future_job.log"
echo "Within-year job log: artifacts/care-litelocal-p31-diagnostics-v1/within_year_job.log"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
