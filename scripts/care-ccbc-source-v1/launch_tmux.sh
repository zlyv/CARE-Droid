#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${CCBC_PROJECT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
base="$PROJECT/outputs/care-ccbc-source-v1"
output="$base"
version=1
while [[ -e "$output" ]] && find "$output" -mindepth 1 -print -quit 2>/dev/null | grep -q .; do
  version=$((version+1)); output="$PROJECT/outputs/care-ccbc-source-v${version}"
done
session="care-ccbc-source-v${version}"
console="$PROJECT/outputs/care-ccbc-source-v${version}.console.log"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "BLOCKED_BY_CONTRACT: tmux session already exists: $session"; exit 2
fi
tmux new-session -d -s "$session" \
  "cd '$PROJECT' && CCBC_PROJECT='$PROJECT' CCBC_OUTPUT='$output' bash '$SCRIPT_DIR/run_on_server.sh' > '$console' 2>&1"
echo "Started: $session"
echo "Output: $output"
echo "Monitor: tail -F $console"
echo "Run log: tail -F $output/logs/run.log"
echo "Attach: tmux attach -t $session"
echo "Detach: Ctrl-b, then d"
