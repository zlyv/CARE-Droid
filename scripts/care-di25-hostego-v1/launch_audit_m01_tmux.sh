#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
SESSION=care-di25-hostego-audit-m01
LOG=artifacts/care-di25-hostego-v1.audit-m01.console.log
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 1
fi
tmux new-session -d -s "$SESSION" "cd /home/boyuanjie/projects/tasa-droid && bash scripts/care-di25-hostego-v1/run_audit_m01_only.sh 2>&1 | tee '$LOG'"
echo "Started audit-only session: $SESSION"
echo "Monitor: tail -F $LOG"
