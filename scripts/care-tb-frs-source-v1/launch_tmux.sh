#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAME="care-tb-frs-source-v1"; OUT="$ROOT/outputs/care-tb-frs-source-v1"; CONSOLE="$ROOT/outputs/care-tb-frs-source-v1.console.log"
mkdir -p "$ROOT/outputs"
if tmux has-session -t "$NAME" 2>/dev/null; then echo "Session already exists: $NAME"; exit 1; fi
tmux new-session -d -s "$NAME" "cd '$ROOT' && bash scripts/care-tb-frs-source-v1/run_on_server.sh 2>&1 | tee -a '$CONSOLE'"
echo "Started: $NAME"; echo "Monitor: tail -F $CONSOLE"; echo "Attach: tmux attach -t $NAME"; echo "Detach: Ctrl-b, then d"
