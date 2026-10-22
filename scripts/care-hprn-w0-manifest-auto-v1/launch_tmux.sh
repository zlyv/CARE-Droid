#!/usr/bin/env bash
set -euo pipefail
SESSION=care_hprn_w0_manifest_auto_v1
ROOT=/home/boyuanjie/projects/tasa-droid
LOG="$ROOT/outputs/care_hprn_w0_manifest_auto_v1.console.log"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash scripts/care-hprn-w0-manifest-auto-v1/run.sh > '$LOG' 2>&1"
echo "$SESSION"
echo "$LOG"
