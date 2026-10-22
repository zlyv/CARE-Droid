#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION="care_hostego_seed"
LOG="artifacts/care-di25-hostego-v1/seed_robustness.console.log"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi
mkdir -p artifacts/care-di25-hostego-v1/seed_robustness
tmux new-session -d -s "$SESSION" "cd /home/boyuanjie/projects/tasa-droid && CARE_ML_PYTHON=/home/boyuanjie/envs/tif/bin/python /home/boyuanjie/envs/tif/bin/python scripts/care-di25-hostego-v1/run_seed_robustness.py --mode run --replicate all --device auto 2>&1 | tee '$LOG'; test \${PIPESTATUS[0]} -eq 0 && /home/boyuanjie/envs/tif/bin/python scripts/care-di25-hostego-v1/summarize_seed_robustness.py 2>&1 | tee -a '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Attach: tmux attach -t $SESSION"
