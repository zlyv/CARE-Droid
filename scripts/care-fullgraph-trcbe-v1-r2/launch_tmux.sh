#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${FULLGRAPH_PROJECT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SESSION="${FULLGRAPH_TMUX_SESSION:-care-fullgraph-trcbe-v2}"
OUTPUT="${FULLGRAPH_OUTPUT:-$PROJECT/outputs/care-fullgraph-trcbe-v2}"
CONSOLE="${FULLGRAPH_CONSOLE:-$PROJECT/outputs/care-fullgraph-trcbe-v2.console.log}"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "BLOCKED_BY_CONTRACT tmux session already exists: $SESSION"; exit 2; fi
mkdir -p "$(dirname "$CONSOLE")"
tmux new-session -d -s "$SESSION" "cd '$PROJECT' && FULLGRAPH_PROJECT='$PROJECT' FULLGRAPH_OUTPUT='$OUTPUT' bash '$SCRIPT_DIR/run_on_server.sh' > '$CONSOLE' 2>&1"
echo "Started: $SESSION"
echo "Output: $OUTPUT"
echo "Monitor: tail -F $CONSOLE"
echo "Run log: tail -F $OUTPUT/logs/run.log"
echo "Attach: tmux attach -t $SESSION"
