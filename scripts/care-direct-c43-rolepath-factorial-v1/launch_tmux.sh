#!/usr/bin/env bash
set -euo pipefail
PROJECT="${RPF_PROJECT:-/home/boyuanjie/projects/tasa-droid}";SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"&&pwd)";SESSION="care-direct-c43-rolepath-v1";CONSOLE="$PROJECT/outputs/care-direct-c43-rolepath-factorial-v1.console.log"
mkdir -p "$PROJECT/outputs";if tmux has-session -t "$SESSION" 2>/dev/null;then echo "Session already exists: $SESSION";exit 1;fi
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && RPF_GPU='${RPF_GPU:-1}' bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee '$CONSOLE'"
echo "Started: $SESSION";echo "Monitor: tail -F $CONSOLE";echo "Worker logs: tail -F $PROJECT/outputs/care-direct-c43-rolepath-factorial-v1/logs/*.log";echo "Attach: tmux attach -t $SESSION"
