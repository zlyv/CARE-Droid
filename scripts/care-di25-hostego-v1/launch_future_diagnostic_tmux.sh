#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION="care_hostego_future_diag"
LOG="artifacts/care-di25-hostego-v1/diagnostic_future.console.log"
SEED_REPORT="artifacts/care-di25-hostego-v1/seed_robustness/SEED_ROBUSTNESS_REPORT.txt"
if [[ ! -f "$SEED_REPORT" ]]; then
  echo "BLOCKED_BY_CONTRACT: seed robustness summary must finish first: $SEED_REPORT" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi
mkdir -p artifacts/care-di25-hostego-v1/diagnostic_future
tmux new-session -d -s "$SESSION" "cd /home/boyuanjie/projects/tasa-droid && /home/boyuanjie/envs/tif/bin/python scripts/care-di25-hostego-v1/run_temporal_diagnostic.py --preflight-only 2>&1 | tee '$LOG'; test \${PIPESTATUS[0]} -eq 0 && /home/boyuanjie/envs/tif/bin/python scripts/care-di25-hostego-v1/run_temporal_diagnostic.py 2>&1 | tee -a '$LOG'; test \${PIPESTATUS[0]} -eq 0 && /home/boyuanjie/envs/tif/bin/python scripts/care-di25-hostego-v1/summarize_temporal_diagnostic.py 2>&1 | tee -a '$LOG'; test \${PIPESTATUS[0]} -eq 0 && /home/boyuanjie/envs/tif/bin/python scripts/care-di25-hostego-v1/audit_final_protocol.py --readiness-only 2>&1 | tee -a '$LOG'; test \${PIPESTATUS[0]} -eq 0 && /home/boyuanjie/envs/tif/bin/python scripts/care-di25-hostego-v1/final_report.py 2>&1 | tee -a '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Attach: tmux attach -t $SESSION"
