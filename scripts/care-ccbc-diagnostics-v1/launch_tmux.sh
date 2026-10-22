#!/usr/bin/env bash
set -euo pipefail
PROJECT="${CCBC_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${CCBC_OUTPUT:-$PROJECT/outputs/care-ccbc-diagnostics-v1}"
SESSION="${CCBC_SESSION:-care-ccbc-diagnostics-v1}"
GPU_FROZEN="${CCBC_GPU_FROZEN:-1}"
GPU_WITHIN="${CCBC_GPU_WITHIN:-1}"
CONSOLE="$PROJECT/outputs/care-ccbc-diagnostics-v1.console.log"
mkdir -p "$PROJECT/outputs" "$OUT/logs"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Session exists: $SESSION"; exit 1; fi
tmux new-session -d -s "$SESSION" \
  "cd '$PROJECT' && CCBC_OUTPUT='$OUT' CCBC_SOURCE_OUTPUT='$PROJECT/outputs/care-ccbc-source-v3' CCBC_GPU_FROZEN='$GPU_FROZEN' CCBC_GPU_WITHIN='$GPU_WITHIN' bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee -a '$CONSOLE'"
echo "Started: $SESSION"
echo "Output: $OUT"
echo "GPU plan: frozen=$GPU_FROZEN within=$GPU_WITHIN (same GPU means sequential execution)"
echo "Monitor: tail -F $CONSOLE"
echo "Frozen GPU log: tail -F $OUT/logs/frozen_attention.log"
echo "Within GPU log: tail -F $OUT/logs/within_attention.log"
echo "Attach: tmux attach -t $SESSION"
