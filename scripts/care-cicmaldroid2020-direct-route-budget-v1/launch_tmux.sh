#!/usr/bin/env bash
set -euo pipefail
PROJECT="${CIC_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
NAME="care-cicmaldroid-direct-budget-v1"
OUTPUT="${CIC_OUTPUT:-$PROJECT/outputs/care-cicmaldroid2020-direct-route-budget-v1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$PROJECT/outputs" "$OUTPUT/logs"
if tmux has-session -t "$NAME" 2>/dev/null; then
  echo "Session already exists: $NAME"; exit 1
fi
tmux new-session -d -s "$NAME" "cd '$PROJECT' && bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee -a '$PROJECT/outputs/care-cicmaldroid2020-direct-route-budget-v1.console.log'"
echo "Started: $NAME"
echo "GPU: ${CIC_GPU:-1} (override with CIC_GPU=0 before launch)"
echo "Monitor: tail -F $PROJECT/outputs/care-cicmaldroid2020-direct-route-budget-v1.console.log"
echo "Attach: tmux attach -t $NAME"

