#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${CCBC_PROJECT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
raw_output="${1:-$PROJECT/outputs/care-ccbc-source-v3}"
OUTPUT="$(realpath -m "$raw_output")"
case "$OUTPUT" in
  "$(realpath -m "$PROJECT")"/outputs/care-ccbc-source-v*) ;;
  *) echo "BLOCKED_BY_CONTRACT: unsafe resume output=$OUTPUT"; exit 2 ;;
esac
if [[ ! -d "$OUTPUT" || ! -f "$OUTPUT/cache/formal_attention.npz" ]]; then
  echo "BLOCKED_BY_CONTRACT: resume output lacks validated attention cache: $OUTPUT"; exit 2
fi
base="$(basename "$OUTPUT")"
session="${base}-resume"
console="$PROJECT/outputs/${base}.console.log"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "BLOCKED_BY_CONTRACT: tmux session already exists: $session"; exit 2
fi
tmux new-session -d -s "$session" \
  "cd '$PROJECT' && CCBC_PROJECT='$PROJECT' CCBC_OUTPUT='$OUTPUT' bash '$SCRIPT_DIR/run_on_server.sh' >> '$console' 2>&1"
echo "Resumed: $session"
echo "Output: $OUTPUT"
echo "Monitor: tail -F $console"
echo "Run log: tail -F $OUTPUT/logs/run.log"
echo "Attach: tmux attach -t $session"
