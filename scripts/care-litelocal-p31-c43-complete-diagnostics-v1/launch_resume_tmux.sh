#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-p31-c43-complete-resume-v1
LOG=artifacts/care-litelocal-p31-c43-complete-diagnostics-v1.resume.console.log
tmux has-session -t "$SESSION" 2>/dev/null && { echo "Already running: $SESSION"; exit 0; }
tmux new-session -d -s "$SESSION" "cd /home/boyuanjie/projects/tasa-droid && bash scripts/care-litelocal-p31-c43-complete-diagnostics-v1/resume_2020_2021.sh 2>&1 | tee $LOG"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "GPU log: tail -F artifacts/care-litelocal-p31-c43-complete-diagnostics-v1/logs/within_2020_2021.gpu0.log"
echo "Attach: tmux attach -t $SESSION"
