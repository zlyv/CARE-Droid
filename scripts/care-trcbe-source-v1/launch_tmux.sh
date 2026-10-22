#!/usr/bin/env bash
set -euo pipefail
PROJECT="${TRCBE_PROJECT:-/home/boyuanjie/projects/tasa-droid}";SCRIPT_DIR="$(cd "$(dirname "$0")"&&pwd)";SESSION="${TRCBE_SESSION:-care-trcbe-source-v1}"
version=1
while true;do OUT="$PROJECT/outputs/care-trcbe-source-v${version}";CONSOLE="$PROJECT/outputs/care-trcbe-source-v${version}.console.log";[[ ! -e "$OUT" && ! -e "$CONSOLE" ]]&&break;version=$((version+1));done
tmux has-session -t "$SESSION" 2>/dev/null&&{ echo "Session exists: $SESSION";exit 1; }
mkdir -p "$PROJECT/outputs"
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && TRCBE_OUTPUT='$OUT' bash '$SCRIPT_DIR/run_on_server.sh' 2>&1 | tee -a '$CONSOLE'"
echo "Started: $SESSION";echo "Output: $OUT";echo "Monitor: tail -F $CONSOLE";echo "No label probe or model training is run by this package."

