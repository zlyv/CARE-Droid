#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/boyuanjie/projects/tasa-droid"
SESSION="care-needgate-source-v1"
LOG="$PROJECT_ROOT/artifacts/care-di25-noego-needgate-source-v1.console.log"
PACKAGE="$PROJECT_ROOT/scripts/care-di25-noego-needgate-source-v1"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/artifacts"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "BLOCKED_BY_CONTRACT: tmux session already exists: $SESSION" >&2
  exit 2
fi

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "1" ]]; then
  echo "BLOCKED_BY_CONTRACT: launch with CUDA_VISIBLE_DEVICES=1" >&2
  exit 2
fi

# Physical GPU 1 is deliberately propagated into the detached shell and is
# addressed inside that process as cuda:0.
tmux new-session -d -s "$SESSION" \
  "cd '$PROJECT_ROOT' && CUDA_VISIBLE_DEVICES=1 CARE_DEVICE=cuda:0 bash '$PACKAGE/run_formal_on_server.sh' 2>&1 | tee '$LOG'"

echo "Started: $SESSION"
echo "Monitor: tail -F $LOG"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b, then d"
