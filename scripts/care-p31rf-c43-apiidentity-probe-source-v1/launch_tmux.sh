#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION=care-p31rf-apiidentity-v1
LOG="$PROJECT/artifacts/care-p31rf-c43-apiidentity-probe-source-v1.console.log"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && bash scripts/care-p31rf-c43-apiidentity-probe-source-v1/run_on_server.sh 2>&1 | tee '$LOG'"
echo "Started: $SESSION"
echo "Monitor: tail -F artifacts/care-p31rf-c43-apiidentity-probe-source-v1.console.log"
echo "Attach: tmux attach -t $SESSION"

