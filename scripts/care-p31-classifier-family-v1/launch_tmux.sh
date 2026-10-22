#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PCF_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${PCF_OUTPUT:-$PROJECT/outputs/care-p31-classifier-family-v1}"
SESSION="care-p31-classifier-family-v1"
mkdir -p "$PROJECT/outputs" "$OUTPUT/logs"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Session already exists: $SESSION"; exit 1; fi
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && bash scripts/care-p31-classifier-family-v1/run_on_server.sh >> '$PROJECT/outputs/care-p31-classifier-family-v1.console.log' 2>&1"
echo "Started: $SESSION"
echo "Monitor: tail -F $PROJECT/outputs/care-p31-classifier-family-v1.console.log"
echo "Run log: tail -F $OUTPUT/logs/run.log"

