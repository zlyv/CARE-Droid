#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/boyuanjie/projects/tasa-droid"
SESSION="care-noego-needgate-v1"
LOG="${PROJECT_ROOT}/artifacts/care-di25-noego-needgate-probe-v1.console.log"

cd "${PROJECT_ROOT}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  echo "Attach: tmux attach -t ${SESSION}"
  exit 2
fi
mkdir -p "${PROJECT_ROOT}/artifacts"
tmux new-session -d -s "${SESSION}" \
  "cd '${PROJECT_ROOT}' && bash scripts/care-di25-noego-needgate-probe-v1/run_on_server.sh 2>&1 | tee '${LOG}'"
echo "Started: ${SESSION}"
echo "Monitor: tail -F ${LOG}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Detach: Ctrl-b, then d"
