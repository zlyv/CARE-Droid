#!/usr/bin/env bash
set -euo pipefail
PROJECT="${CSHM_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
NAME="care-signed-host-mil-v1"
OUTPUT="${CSHM_OUTPUT:-$PROJECT/outputs/care-signed-host-mil-v1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$PROJECT/outputs" "$OUTPUT/logs"
if tmux has-session -t "$NAME" 2>/dev/null; then echo "Session already exists: $NAME"; exit 1; fi
GPU="${CSHM_GPU:-1}"
tmux new-session -d -s "$NAME" "cd '$PROJECT' && CUDA_VISIBLE_DEVICES='$GPU' bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee -a '$PROJECT/outputs/care-signed-host-mil-v1.console.log'"
echo "Started: $NAME"
echo "GPU: $GPU"
echo "Monitor: tail -F $PROJECT/outputs/care-signed-host-mil-v1.console.log"
echo "Attach: tmux attach -t $NAME"

